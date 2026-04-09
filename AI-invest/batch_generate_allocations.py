"""Two-stage batch asset allocation: generate proposals then extract weights."""

import csv
import hashlib
import json
import os
import re
import time
from datetime import datetime, timezone

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# --- Config ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_PATH = os.path.join(BASE_DIR, "市场分析-提示词模板.md")
CSV_PATH = os.path.join(BASE_DIR, "..", "420", "420_growth_clients_35_minimal.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

STAGE1_MODEL = "qwen3-235b-a22b-instruct-2507"
STAGE2_MODEL = "qwen3.6-plus"
MAX_RETRIES = 3
RETRY_DELAY = 5
MIN_INTERVAL = 2  # seconds between API calls

# Eligibility rules
ALLOWED_ASSETS = {
    1: ["CASH", "BOND"],
    2: ["CASH", "BOND", "ALT"],
    3: ["CASH", "BOND", "EQUITY", "ALT"],
    4: ["CASH", "BOND", "EQUITY", "ALT"],
    5: ["CASH", "BOND", "EQUITY", "ALT"],
}

STAGE1_USER_PROMPT = "请结合我当前的情况和市场环境，给我一个专属的资产配置方案。"

STAGE2_SYSTEM_PROMPT = """你是一个结构化数据提取助手。

任务：从文本中提取"最终推荐资产配置方案"的四大类资产权重。

输出规则：
1. 只输出一个合法的裸 JSON 对象
2. 不要输出任何解释、markdown、代码块、前后缀或注释
3. JSON 中必须始终包含四个键：CASH, BOND, EQUITY, ALT
4. 每个值必须是数字或 null
5. 数字必须是 0 到 100 之间的百分比数值
6. 不要输出 0.7，必须输出 70
7. 不要输出 "70%"，必须输出 70
8. 不要输出字符串形式的数字，如 "70"
9. 若某类资产未配置、明确不可投或权重为 0，则输出 0
10. 若无法确定最终推荐方案，则必须输出：
{{"CASH":null,"BOND":null,"EQUITY":null,"ALT":null}}"""

STAGE2_USER_TEMPLATE = """请从下面这段资产配置方案中，提取"最终推荐资产配置方案"的四大类资产权重。

识别规则：
1. 文中可能有多个备选方案、情景方案、中间方案，你只能提取最终采用的那一组
2. 优先识别以下明确的最终方案锚点：
   - 最终推荐资产配置方案
   - 最终推荐资产配置权重
   - 最终方案
   - 最终推荐
3. 如果文中出现以下表达，也视为最终方案线索：
   - 推荐采用某方案
   - 优先采纳某方案
   - 主推方案
   - 默认策略
   - 综合建议后的最终配置
   - 落地配置建议
4. 如果原文先说明"最终采用的是某个方案名"，你需要回到该方案对应的权重表中提取四类权重
5. 不要提取备选方案、情景方案、中间推演方案，除非原文明确说最终采用该方案
6. 如果没有明确标题或方案名，就提取全文结论部分最明确的一组四大类权重
7. 未配置、明确不可投或明确为 0 的资产，输出 0
8. 严格遵守原文，不要脑补，不要自行改写

只输出裸 JSON，例如：
{{"CASH":70,"BOND":30,"EQUITY":0,"ALT":0}}

文本如下：
{raw_output}"""


def load_template():
    with open(TEMPLATE_PATH, "r", encoding="utf-8") as f:
        return f.read()


def load_clients():
    rows = []
    with open(CSV_PATH, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def build_system_prompt(template: str, risk_level: int, lifecycle: str) -> str:
    """Replace the client profile JSON block in the template."""
    allowed = ALLOWED_ASSETS[risk_level]
    new_json = json.dumps(
        {
            "client_risk_tolerance": f"C{risk_level}",
            "investment_objectives": "增值",
            "life_stage": lifecycle,
            "allowed_asset_classes": allowed,
        },
        ensure_ascii=False,
        indent=2,
    )
    # Replace the JSON block between ```json and ``` under "客户投资前提"
    pattern = r'(# 客户投资前提\s*\n```json\s*\n)(\{.*?\})(\s*\n```)'
    replacement = rf'\g<1>{new_json}\g<3>'
    result = re.sub(pattern, replacement, template, flags=re.DOTALL)
    return result


class APIClient:
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        if not api_key:
            raise ValueError("Set OPENAI_API_KEY or DASHSCOPE_API_KEY in .env")
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self._last_call = 0.0

    def chat(self, model: str, system_prompt: str, user_message: str,
             temperature: float = 0.1, max_tokens: int = 8192) -> dict:
        """Call API with retry. Returns dict with raw_output, finish_reason."""
        elapsed = time.time() - self._last_call
        if elapsed < MIN_INTERVAL:
            time.sleep(MIN_INTERVAL - elapsed)

        last_error = None
        for attempt in range(MAX_RETRIES):
            try:
                self._last_call = time.time()
                resp = self.client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message},
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                text = resp.choices[0].message.content or ""
                # Strip <think>...</think> blocks (Qwen reasoning)
                text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
                finish = resp.choices[0].finish_reason or "unknown"
                return {"raw_output": text, "finish_reason": finish}
            except Exception as e:
                last_error = e
                if attempt < MAX_RETRIES - 1:
                    wait = RETRY_DELAY * (attempt + 1)
                    print(f"  [retry {attempt+1}/{MAX_RETRIES}] {e}, waiting {wait}s...")
                    time.sleep(wait)

        raise RuntimeError(f"API failed after {MAX_RETRIES} retries: {last_error}")


def strip_code_fence(text: str) -> str:
    """Remove ```json ... ``` or ``` ... ``` code fences, return inner content."""
    text = text.strip()
    # Match ```json\n...\n``` or ```\n...\n```
    m = re.search(r'```(?:json)?\s*\n?(.*?)\n?\s*```', text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return text


def extract_first_json_object(text: str):
    """Extract the first { ... } block from text. Handles nested braces."""
    start = text.find('{')
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def normalize_weight_value(val):
    """Convert a single weight value to float. Handles '70%', '70', 0.7, 70."""
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        val = val.strip().rstrip('%').strip()
        try:
            return float(val)
        except ValueError:
            return None
    return None


def normalize_allocation_dict(data: dict) -> dict:
    """Normalize {CASH, BOND, EQUITY, ALT} values: handle %, 0~1 decimals, strings."""
    keys = ["CASH", "BOND", "EQUITY", "ALT"]
    vals = {}
    for k in keys:
        vals[k] = normalize_weight_value(data.get(k))

    # Check if values look like 0~1 ratios (sum ≈ 1) and scale to 100
    parseable = [v for v in vals.values() if v is not None]
    if len(parseable) >= 2:
        s = sum(parseable)
        if 0.99 <= s <= 1.01:
            # All values are 0~1 ratios, scale to percentages
            for k in keys:
                if vals[k] is not None:
                    vals[k] = round(vals[k] * 100, 2)

    # Round to 2 decimal places
    for k in keys:
        if vals[k] is not None:
            vals[k] = round(vals[k], 2)

    return vals


def parse_weights(stage2_output: str) -> dict:
    """Parse CASH/BOND/EQUITY/ALT from stage2 output. Returns dict with weights + parse_status."""
    result = {"CASH": None, "BOND": None, "EQUITY": None, "ALT": None,
              "weight_sum": None, "parse_status": "failed"}

    text = stage2_output.strip()
    if not text:
        return result

    # Step 1: strip code fence
    cleaned = strip_code_fence(text)

    # Step 2: try parsing cleaned text as JSON
    parsed = None
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Step 3: fallback — extract first JSON object from raw text
    if not parsed:
        json_str = extract_first_json_object(text)
        if json_str:
            try:
                parsed = json.loads(json_str)
            except json.JSONDecodeError:
                pass

    if not parsed or not isinstance(parsed, dict):
        return result

    # Step 4: normalize values (handle %, 0~1, strings)
    normalized = normalize_allocation_dict(parsed)

    for k in ["CASH", "BOND", "EQUITY", "ALT"]:
        result[k] = normalized[k]

    # Step 5: compute sum and determine status
    vals = [result[k] for k in ["CASH", "BOND", "EQUITY", "ALT"] if result[k] is not None]
    if len(vals) == 4:
        result["weight_sum"] = round(sum(vals), 2)
        if 99.5 <= result["weight_sum"] <= 100.5:
            result["parse_status"] = "success"
        else:
            result["parse_status"] = "sum_not_100"
    else:
        result["parse_status"] = "failed"

    return result


def run():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    template = load_template()
    clients = load_clients()
    template_hash = hashlib.md5(template.encode()).hexdigest()[:12]
    api = APIClient()

    print(f"Loaded {len(clients)} clients, template hash={template_hash}")

    # --- Stage 1 ---
    stage1_path = os.path.join(OUTPUT_DIR, "stage1_raw_generations_v3.jsonl")
    stage1_results = []

    print("\n=== Stage 1: Generating allocation proposals ===")
    for i, row in enumerate(clients):
        client_id = int(row["id"])
        lifecycle = row["lifecycle"]
        risk_level = int(row["risk_level"])
        allowed = ALLOWED_ASSETS[risk_level]

        print(f"[{i+1}/35] id={client_id} lifecycle={lifecycle} risk=C{risk_level} ...", end=" ", flush=True)

        sys_prompt = build_system_prompt(template, risk_level, lifecycle)

        try:
            resp = api.chat(STAGE1_MODEL, sys_prompt, STAGE1_USER_PROMPT, temperature=0.7, max_tokens=8192)
            status = "success"
        except Exception as e:
            resp = {"raw_output": "", "finish_reason": "error"}
            status = f"error: {e}"
            print(f"FAILED: {e}")

        record = {
            "id": client_id,
            "lifecycle": lifecycle,
            "risk_level": risk_level,
            "allowed_asset_classes": allowed,
            "stage1_model": STAGE1_MODEL,
            "system_prompt_path": TEMPLATE_PATH,
            "system_prompt_hash": template_hash,
            "user_prompt": STAGE1_USER_PROMPT,
            "raw_output": resp["raw_output"],
            "finish_reason": resp["finish_reason"],
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": status,
        }
        stage1_results.append(record)
        print(f"OK ({len(resp['raw_output'])} chars)")

    with open(stage1_path, "w", encoding="utf-8") as f:
        for r in stage1_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nStage 1 saved to {stage1_path}")

    # --- Stage 2 ---
    stage2_path = os.path.join(OUTPUT_DIR, "stage2_extracted_weights_v3.jsonl")
    stage2_results = []

    print("\n=== Stage 2: Extracting weights ===")
    for i, s1 in enumerate(stage1_results):
        client_id = s1["id"]
        print(f"[{i+1}/35] id={client_id} ...", end=" ", flush=True)

        if not s1["raw_output"]:
            print("SKIP (empty stage1)")
            stage2_results.append({
                "id": client_id, "stage2_model": STAGE2_MODEL,
                "stage1_raw_output": "", "stage2_raw_output": "",
                "CASH": None, "BOND": None, "EQUITY": None, "ALT": None,
                "weight_sum": None, "parse_status": "failed",
                "created_at": datetime.now(timezone.utc).isoformat(),
            })
            continue

        user_msg = STAGE2_USER_TEMPLATE.format(raw_output=s1["raw_output"])

        try:
            resp = api.chat(STAGE2_MODEL, STAGE2_SYSTEM_PROMPT, user_msg, temperature=0.0, max_tokens=512)
            s2_output = resp["raw_output"]
        except Exception as e:
            s2_output = ""
            print(f"FAILED: {e}")

        weights = parse_weights(s2_output)
        record = {
            "id": client_id,
            "stage2_model": STAGE2_MODEL,
            "stage1_raw_output": s1["raw_output"],
            "stage2_raw_output": s2_output,
            **{k: weights[k] for k in ["CASH", "BOND", "EQUITY", "ALT", "weight_sum", "parse_status"]},
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        stage2_results.append(record)
        print(f"OK  CASH={weights['CASH']} BOND={weights['BOND']} EQ={weights['EQUITY']} ALT={weights['ALT']} sum={weights['weight_sum']} [{weights['parse_status']}]")

    with open(stage2_path, "w", encoding="utf-8") as f:
        for r in stage2_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nStage 2 saved to {stage2_path}")

    # --- Final CSV ---
    csv_path = os.path.join(OUTPUT_DIR, "extracted_weights_v3.csv")
    fieldnames = ["id", "lifecycle", "risk_level", "cash_pct", "bond_pct",
                  "equity_pct", "commodity_pct", "CASH", "BOND", "EQUITY", "ALT",
                  "weight_sum", "parse_status"]

    # Build lookup from original CSV
    client_map = {int(r["id"]): r for r in clients}

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s2 in stage2_results:
            cid = s2["id"]
            orig = client_map[cid]
            writer.writerow({
                "id": cid,
                "lifecycle": orig["lifecycle"],
                "risk_level": orig["risk_level"],
                "cash_pct": orig["cash_pct"],
                "bond_pct": orig["bond_pct"],
                "equity_pct": orig["equity_pct"],
                "commodity_pct": orig["commodity_pct"],
                "CASH": s2["CASH"],
                "BOND": s2["BOND"],
                "EQUITY": s2["EQUITY"],
                "ALT": s2["ALT"],
                "weight_sum": s2["weight_sum"],
                "parse_status": s2["parse_status"],
            })
    print(f"Final CSV saved to {csv_path}")

    # Summary
    success = sum(1 for r in stage2_results if r["parse_status"] == "success")
    failed = sum(1 for r in stage2_results if r["parse_status"] == "failed")
    not100 = sum(1 for r in stage2_results if r["parse_status"] == "sum_not_100")
    print(f"\nDone. success={success} sum_not_100={not100} failed={failed}")


if __name__ == "__main__":
    run()
