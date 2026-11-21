import copy
import math
from typing import Any, Dict, List


AGGREGATIONS = frozenset({"count", "sum", "avg", "min", "max", "ratio"})


def aggregate(rows: List[Dict[str, Any]], spec: Dict[str, Any]) -> Any:
    operation = spec.get("aggregation")
    if operation not in AGGREGATIONS:
        raise ValueError(f"Unsupported aggregation: {operation}")
    selected = [r for r in rows if all(r.get(k) == v for k, v in (spec.get("where") or {}).items())]
    if operation == "count":
        return len(selected)
    def numbers(field: str) -> List[float]:
        if not field:
            raise ValueError("Metric requires a source field")
        values = []
        for row in selected:
            value = row.get(field)
            if value is None or value == "":
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"Metric source field {field} must be numeric")
            values.append(float(value))
        return values
    if operation == "ratio":
        numerator = numbers(spec.get("numeratorField", ""))
        denominator = numbers(spec.get("denominatorField", ""))
        total = sum(denominator)
        scale = spec.get("scale", 1)
        if not isinstance(scale, (int, float)) or isinstance(scale, bool) or not math.isfinite(scale):
            raise ValueError("Metric scale must be finite")
        return sum(numerator) / total * scale if total and numerator else None
    values = numbers(spec.get("field", ""))
    if not values:
        return None
    if operation == "sum":
        return sum(values)
    if operation == "avg":
        return sum(values) / len(values)
    return min(values) if operation == "min" else max(values)


def compute_definition(definition: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(definition)
    bag = out.get("data") if isinstance(out.get("data"), dict) else out.get("mock", {})
    tables = {b["dataKey"]: b for p in out.get("pages", []) for b in p.get("blocks", []) if b.get("type") == "dataTable"}
    for page in out.get("pages", []):
        for block in page.get("blocks", []):
            source = block.get("sourceDataKey")
            if not source:
                continue
            if source not in tables or not isinstance(bag.get(source), list):
                raise ValueError(f"Unknown metric source dataset: {source}")
            rows = bag[source]
            fields = {c["field"] for c in tables[source]["columns"]}
            specs = block.get("items", []) if block["type"] == "kpiRow" else block.get("series", [])
            for spec in specs:
                referenced = {spec.get("field"), spec.get("numeratorField"), spec.get("denominatorField")} - {None, ""}
                referenced.update((spec.get("where") or {}).keys())
                if not referenced.issubset(fields):
                    raise ValueError(f"Metric references unknown fields: {referenced - fields}")
            if block["type"] == "kpiRow":
                bag[block["dataKey"]] = {s["valueKey"]: aggregate(rows, s) for s in specs}
            elif block["type"] in {"lineChart", "barChart"}:
                group_field = block.get("groupBy")
                if group_field not in fields:
                    raise ValueError("Chart groupBy must reference a source column")
                groups: Dict[str, List[Dict[str, Any]]] = {}
                for row in rows:
                    value = row.get(group_field)
                    if value is not None:
                        groups.setdefault(str(value), []).append(row)
                bag[block["dataKey"]] = [
                    {block["xAxisKey"]: key, **{s["dataKey"]: aggregate(group, s) for s in specs}}
                    for key, group in sorted(groups.items())
                ]
            else:
                raise ValueError("sourceDataKey is supported only on KPI and line/bar chart blocks")
    return out