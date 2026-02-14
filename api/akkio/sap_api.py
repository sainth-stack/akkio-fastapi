from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any
from urllib.parse import quote
import json
import re
import requests

from .explore_functions.llm import call_llm_with_usage

sap_router = APIRouter()

SAP_BASE_URL = "https://seleccionapidev.test02.apimanagement.us10.hana.ondemand.com/odata"
SAP_USERNAME = "abaphana82"
SAP_PASSWORD = "welcome@82"


class SapQueryRequest(BaseModel):
    input: str
    email: Optional[str] = None


SAP_BTP_API = "https://ans-webhook-happy-civet-oc.cfapps.us10-001.hana.ondemand.com/api/history"
SAP_BATCH_API = "https://tg-monitoring-backend.cfapps.us10-001.hana.ondemand.com/v1/automate/idoc-data/status"


class SapExternalQueryRequest(BaseModel):
    input: str
    mode: str  # "btp" or "batch"
    email: Optional[str] = None


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None

    try:
        return json.loads(match.group(0))
    except Exception:
        return None


def _build_odata_url(intent: Dict[str, Any]) -> str:
    service = intent.get("service") or intent.get("odataService")
    entity = intent.get("entity") or intent.get("odataEntity")
    value = intent.get("value")
    top = intent.get("$top") or intent.get("top")
    filter_expr = intent.get("$filter") or intent.get("filter")

    if not service or not entity:
        raise HTTPException(status_code=400, detail="Could not resolve SAP service/entity from input.")

    url = f"{SAP_BASE_URL}/{service}/{entity}"

    if value is not None and str(value).strip() not in ("", "0"):
        url += f"('{quote(str(value).strip())}')"

    query_params = ["$format=json"]
    if top:
        top_text = str(top).strip()
        top_match = re.search(r"(\d+)", top_text)
        if top_match:
            query_params.append(f"$top={top_match.group(1)}")
    if filter_expr:
        query_params.append("$filter=" + quote(str(filter_expr).strip(), safe=" ()=,'/-"))

    return f"{url}?{'&'.join(query_params)}"


@sap_router.post("/sap/query")
async def query_sap(request: SapQueryRequest):
    user_input = (request.input or "").strip()
    if not user_input:
        raise HTTPException(status_code=400, detail="input is required")

    parser_prompt = """
You are an enterprise integration assistant.

Analyze the user request and identify the correct STANDARD SAP OData service and entity.
Sales Order <-> API_SALES_ORDER_SRV <-> A_SalesOrder
Sales Order Item <-> API_SALES_ORDER_SRV <-> A_SalesOrderItem
Purchase Order <-> API_PURCHASEORDER_PROCESS_SRV <-> A_PurchaseOrder
If other than sales and purchase orders, get the relevant standard SAP OData and fetch the details.

Rules:
- If the request asks for a single entity (e.g. "sales order 4"), set value to the identifier ("4").
- If the request is for a list, set value = "0".
- If no identifier is mentioned, assume a list request.
- Extract filters using valid OData $filter syntax only (do not include "$filter=").
- Extract top as "$top = N" only if explicitly requested.
- If the request is unclear, infer the most likely business entity.
- When multiple entities are possible, prefer Sales Order over Purchase Order.

operation:
- GET_SINGLE for individual entity
- GET_LIST for collections

Return ONLY a valid JSON object with keys:
{
  "service": "...",
  "entity": "...",
  "value": "...",
  "operation": "GET_SINGLE|GET_LIST",
  "$filter": "... or null",
  "$top": "... or null"
}
Do not include markdown or extra text.
""".strip()

    intent: Dict[str, Any]
    try:
        llm_resp = call_llm_with_usage(
            model=None,
            messages=[
                {"role": "system", "content": parser_prompt},
                {"role": "user", "content": user_input},
            ],
            temperature=None,
            email=request.email,
        )
        llm_text = llm_resp.choices[0].message.content if llm_resp and llm_resp.choices else ""
        parsed = _extract_json_object(llm_text)
        if not parsed:
            raise HTTPException(
                status_code=502,
                detail="LLM did not return valid JSON for SAP intent parsing.",
            )
        intent = parsed
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"SAP intent generation failed: {str(exc)}")

    request_url = _build_odata_url(intent)

    try:
        response = requests.get(
            request_url,
            auth=(SAP_USERNAME, SAP_PASSWORD),
            headers={"Accept-Encoding": "application/gzip"},
            timeout=20,
        )
        response.raise_for_status()
        sap_payload = response.json()
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"SAP request failed: {str(exc)}")

    return {
        "input": user_input,
        "response": sap_payload,
    }


@sap_router.post("/sap/query-external")
async def query_sap_external(request: SapExternalQueryRequest):
    """
    For BTP and Batch modes: fetch data from external API, send to LLM with user query,
    return LLM answer.
    """
    user_input = (request.input or "").strip()
    mode = (request.mode or "").lower()
    if not user_input:
        raise HTTPException(status_code=400, detail="input is required")
    if mode not in ("btp", "batch"):
        raise HTTPException(status_code=400, detail="mode must be 'btp' or 'batch'")

    api_url = SAP_BTP_API if mode == "btp" else SAP_BATCH_API

    try:
        resp = requests.get(api_url, timeout=30)
        resp.raise_for_status()
        external_data = resp.json()
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"Failed to fetch from {mode} API: {str(exc)}")

    system_prompt = """You are an assistant that helps users understand and analyze SAP data.

Below is the raw data fetched from an external SAP API. The user has asked a question about this data.

Your task:
1. Analyze the provided data
2. Answer the user's question based on the data
3. If the data is complex (e.g. nested objects, arrays), summarize key insights and answer clearly
4. Use the data to support your answer. If the question cannot be answered from the data, say so
5. Format your response in clear, readable text. Use bullet points or short paragraphs where helpful."""

    user_message = f"""Data from SAP {mode.upper()} API:

```json
{json.dumps(external_data, indent=2, default=str)}
```

User question: {user_input}

Please answer the user's question based on the data above."""

    try:
        llm_resp = call_llm_with_usage(
            model=None,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=None,
            email=request.email,
        )
        answer = (
            llm_resp.choices[0].message.content if llm_resp and llm_resp.choices else ""
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"LLM processing failed: {str(exc)}")

    return {
        "input": user_input,
        "mode": mode,
        "answer": answer,
        "data_source": api_url,
    }
