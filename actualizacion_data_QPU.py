"""
Updates data.csv with the current calibration of IBM QPUs.

Sources:
  api     (default) QiskitRuntimeService: official live data, only the QPUs your account can access.
  scrape  quantum.cloud.ibm.com/computers via Playwright: every public QPU, but breaks if the page layout changes.
  both    API rows, plus scraped rows for QPUs the account cannot see.

If the API source fails (no saved account, no network), it falls back to scraping.

Usage: python actualizacion_data_QPU.py [--source api|scrape|both] [--out data.csv]
"""
import argparse
import statistics
import numpy as np
import pandas as pd

COLUMNS = ["QPU", "Qubits", "Type", "2Q Gate", "2Q Error (Best)", "2Q Error (Layered)",
           "1Q Error", "Readout Error", "T1 (us)", "T2 (us)", "MCPS", "Pending Jobs", "Status"]

def _median_error(target, op_name: str) -> float:
    """Median error of an operation over all its qubits, ignoring broken (error >= 1) entries."""
    if op_name not in target.operation_names:
        return np.nan
    errs = [p.error for p in target[op_name].values() if p is not None and p.error is not None and p.error < 1]
    return statistics.median(errs) if errs else np.nan

def _layered_error(properties) -> float:
    """
    Error per layered gate (EPLG) from the 100-qubit layer fidelity: EPLG = 1 - LF^(1/(N-1)).
    This is what IBM shows as '2Q Error (Layered)'.
    """
    general = {g.name: g.value for g in getattr(properties, "general", []) or []}
    lf_keys = sorted((int(k.split("_")[1]), v) for k, v in general.items() if k.startswith("lf_"))
    if not lf_keys:
        return np.nan
    n, lf = lf_keys[-1]  # longest chain measured (lf_100 on current devices)
    return 1.0 - lf ** (1.0 / (n - 1)) if lf > 0 and n > 1 else np.nan

def backend_to_row(backend) -> dict:
    """Builds one data.csv row from a BackendV2 (real or fake)."""
    target = backend.target
    config = backend.configuration()
    cfg = config.to_dict()

    proc = cfg.get("processor_type") or {}
    qpu_type = f"{proc.get('family', '')} r{proc.get('revision', '')}".strip() if proc else ""

    gate_2q = next((g for g in ("cz", "ecr", "cx") if g in target.operation_names), None)
    errs_2q = [p.error for p in target[gate_2q].values()
               if p is not None and p.error is not None and p.error < 1] if gate_2q else []

    try:
        properties = backend.properties()
    except Exception:
        properties = None

    qprops = [q for q in (target.qubit_properties or []) if q is not None]
    t1 = [q.t1 for q in qprops if q.t1]
    t2 = [q.t2 for q in qprops if q.t2]

    status = backend.status()
    online = status.operational and str(status.status_msg).lower() == "active"
    mcps = cfg.get("mcps")

    return {
        "QPU": backend.name,
        "Qubits": backend.num_qubits,
        "Type": qpu_type,
        "2Q Gate": gate_2q,
        "2Q Error (Best)": min(errs_2q) if errs_2q else np.nan,
        "2Q Error (Layered)": _layered_error(properties) if properties else np.nan,
        "1Q Error": _median_error(target, "sx"),
        "Readout Error": _median_error(target, "measure"),
        "T1 (us)": statistics.median(t1) * 1e6 if t1 else np.nan,
        "T2 (us)": statistics.median(t2) * 1e6 if t2 else np.nan,
        "MCPS": f"{mcps} Hz" if mcps else np.nan,
        "Pending Jobs": status.pending_jobs,
        "Status": "Online" if online else f"Offline {status.status_msg}",
    }

def fetch_from_api(service=None) -> pd.DataFrame:
    """Reads every QPU visible to the saved IBM Quantum account."""
    from qiskit_ibm_runtime import QiskitRuntimeService
    service = service or QiskitRuntimeService()
    rows = [backend_to_row(b) for b in service.backends(simulator=False)]
    return pd.DataFrame(rows, columns=COLUMNS)

def fetch_from_web() -> pd.DataFrame:
    """Scrapes the public QPU table from quantum.cloud.ibm.com (requires playwright)."""
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
                    "Type": cols[6],
                    "2Q Error (Best)": cols[7],
                    "2Q Error (Layered)": cols[8],
                    "Readout Error": cols[9],
                    "MCPS": cols[10],
                    "Status": cols[4]
                })

        browser.close()

    return pd.DataFrame(data).reindex(columns=COLUMNS)

def update_data(source: str = "api", out_path: str = "data.csv") -> pd.DataFrame:
    if source == "scrape":
        df = fetch_from_web()
    else:
        try:
            df = fetch_from_api()
        except Exception as exc:
            print(f"IBM Quantum API unavailable ({exc}); falling back to web scraping.")
            df, source = fetch_from_web(), "scrape"

        if source == "both":
            web = fetch_from_web()
            df = pd.concat([df, web[~web["QPU"].isin(df["QPU"])]], ignore_index=True)

    df.to_csv(out_path, index=False, encoding="utf-8")
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", choices=["api", "scrape", "both"], default="api")
    parser.add_argument("--out", default="data.csv")
    args = parser.parse_args()

    df = update_data(args.source, args.out)
    print(df.to_string(index=False))
