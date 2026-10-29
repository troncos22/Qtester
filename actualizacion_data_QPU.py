import pandas as pd
from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page()
    page.goto("https://quantum.cloud.ibm.com/computers?limit=500")
    page.wait_for_selector("table")

    rows = page.query_selector_all("table tbody tr")
    data = []

    for row in rows:
        cols = [c.inner_text().strip() for c in row.query_selector_all("td")]
        if len(cols) >= 11:
            data.append({
                "QPU": cols[0],
                "Qubits": cols[2],
                "Physical Qbits": cols[3],
                "Tipo": cols[6],
                "2Q Error (Best)": cols[7],
                "2Q Error (Layered)": cols[8],
                "Readout Error": cols[9],
                "MCPS": cols[10],
                "Estado": cols[4]
            })

    browser.close()

# Crear DataFrame y mostrarlo formateado
df = pd.DataFrame(data)
pd.DataFrame(data).to_csv("data.csv", index=False, encoding="utf-8")
print(df.to_string(index=False))

# Opcional: exportar a CSV
# df.to_csv("qpus_ibm.csv", index=False)