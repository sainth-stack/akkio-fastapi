import argparse
import csv
import math
import os
import random
from datetime import datetime, timedelta


def _date_range(start: datetime, periods: int, freq: str):
    cur = start
    for i in range(periods):
        yield cur
        if freq.upper().startswith("D"):
            cur = cur + timedelta(days=1)
        elif freq.upper().startswith("W"):
            cur = cur + timedelta(weeks=1)
        elif freq.upper().startswith("M"):
            # naive month increment
            year = cur.year + (cur.month // 12)
            month = (cur.month % 12) + 1
            cur = cur.replace(year=year, month=month)
        elif freq.upper().startswith("Y"):
            cur = cur.replace(year=cur.year + 1)
        else:
            cur = cur + timedelta(days=1)


def generate_rows(start_date: str, periods: int, freq: str):
    """
    Generate an 8-column synthetic dataset for prediction and forecasting.
    Columns:
      - date (datetime)
      - revenue (float)                [primary target for forecast]
      - marketing_spend (float)
      - price (float)
      - units_sold (int)
      - promotions (int: 0/1)
      - region (category)
      - product_category (category)
    """
    random.seed(42)
    start = datetime.strptime(start_date, "%Y-%m-%d")
    dates = list(_date_range(start, periods, freq))

    # Seasonality and trend helpers
    trend = [0.5 * i for i in range(periods)]  # mild upward trend
    # Seasonal component depending on frequency granularity
    if freq.upper().startswith("D"):
        season = [10 * math.sin(2 * math.pi * i / 30.0) for i in range(periods)]
    elif freq.upper().startswith("W"):
        season = [15 * math.sin(2 * math.pi * i / 12.0) for i in range(periods)]
    elif freq.upper().startswith("M"):
        season = [25 * math.sin(2 * math.pi * i / 12.0) for i in range(periods)]
    else:
        season = [5 * math.sin(2 * math.pi * i / max(1, periods)) for i in range(periods)]

    def noise(std: float) -> float:
        return random.gauss(0.0, std)

    base_revenue = []
    for i in range(periods):
        val = 200 + trend[i] + season[i] + noise(8)
        base_revenue.append(max(val, 20))

    # Marketing spend somewhat correlated with revenue
    marketing_spend = []
    for i in range(periods):
        val = 0.6 * base_revenue[i] + 80 + noise(10)
        marketing_spend.append(max(val, 5))

    # Price has minor variation
    price = []
    for _ in range(periods):
        val = max(random.gauss(20.0, 2.0), 5.0)
        price.append(val)

    # Promotions: more likely during seasonal peaks
    s_min, s_max = min(season), max(season)
    promotions = []
    for i in range(periods):
        p = 0.2 + 0.5 * ((season[i] - s_min) / (s_max - s_min + 1e-6))
        promotions.append(1 if random.random() < p else 0)

    # Units sold inversely related to price, positively related to promotions and marketing_spend
    units_sold = []
    for i in range(periods):
        val = (base_revenue[i] / max(price[i], 1.0)) + 2.5 * promotions[i] + 0.02 * marketing_spend[i] + noise(2.0)
        units_sold.append(max(int(val), 1))

    # Recompute revenue from units_sold and price to tightly couple the target
    revenue = []
    for i in range(periods):
        val = units_sold[i] * price[i] * (1.0 + 0.05 * promotions[i]) + noise(5.0)
        revenue.append(max(val, 10.0))

    regions = ["North", "South", "East", "West"]
    categories = ["A", "B", "C"]

    rows = []
    for i in range(periods):
        region = regions[i % len(regions)]
        product_category = categories[(i // 2) % len(categories)]
        rows.append(
            {
                "date": dates[i].strftime("%Y-%m-%d"),
                "revenue": round(revenue[i], 2),
                "marketing_spend": round(marketing_spend[i], 2),
                "price": round(price[i], 2),
                "units_sold": int(units_sold[i]),
                "promotions": int(promotions[i]),
                "region": region,
                "product_category": product_category,
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic 8-column dataset for prediction and forecasting.")
    parser.add_argument("--start", type=str, default="2024-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--periods", type=int, default=365, help="Number of periods/rows")
    parser.add_argument("--freq", type=str, default="D", help="Frequency: D (days), W (weeks), M (months), Y (years)")
    parser.add_argument("--output", type=str, required=True, help="Output CSV file path")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    rows = generate_rows(args.start, args.periods, args.freq)
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "date",
                "revenue",
                "marketing_spend",
                "price",
                "units_sold",
                "promotions",
                "region",
                "product_category",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()


