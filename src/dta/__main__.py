"""
Punto de entrada para el módulo DTA (Data).
Procesa todos los EDF del directorio DATA_EDF_DIR:
- extrae Airflow y SpO2
- guarda resultados en .npz
- permite reanudar (skip si ya existe output)
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime
from typing import List

from .io_utils import read_channels, save_dta_result
from .config import DATA_EDF_DIR, DTA_RESULTS_DIR


def get_edfs(edf_dir: Path) -> List[Path]:
    """Obtiene la lista ordenada de archivos EDF."""
    return sorted(edf_dir.glob("*.edf"))


def expected_output_path(edf_path: Path, out_dir: Path) -> Path:
    """Ruta esperada del archivo de salida DTA para un EDF."""
    return out_dir / f"{edf_path.stem}_dta_results.npz"


def main() -> None:
    edf_dir = Path(DATA_EDF_DIR)
    out_dir = Path(DTA_RESULTS_DIR)

    if not edf_dir.exists():
        print(f"[ERROR] EDF directory does not exist: {edf_dir.resolve()}", flush=True)
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    edf_files = get_edfs(edf_dir)
    print(f"[INFO] Looking for EDF files in: {edf_dir.resolve()}", flush=True)
    print(f"[INFO] Found {len(edf_files)} EDF files", flush=True)

    processed_ok = 0
    skipped_done = 0
    failed: List[str] = []

    for idx, edf_path in enumerate(edf_files, start=1):
        print(f"\n[{idx}/{len(edf_files)}] Processing: {edf_path.name}", flush=True)

        # Reanudar: saltar si ya existe salida
        save_path_expected = expected_output_path(edf_path, out_dir)
        if save_path_expected.exists():
            skipped_done += 1
            print(f"[INFO] Skipped (already processed): {edf_path.stem}", flush=True)
            continue

        try:
            signals, info = read_channels(edf_path)

            if signals is None:
                err = info.get("error", "unknown error") if isinstance(info, dict) else "unknown error"
                print(f"[WARNING] Skipped → {err}", flush=True)
                failed.append(f"{edf_path}\t{err}")
                continue

            print("[OK] Loaded:", flush=True)
            print(f"   Airflow → {info['airflow_ch']}  (fs={info['fs_airflow']:.1f} Hz)", flush=True)
            print(f"   SpO₂    → {info['spo2_ch']}  (fs={info['fs_spo2']:.1f} Hz)", flush=True)
            print(f"   Duration: {info['duration_s']:.1f} s  |  Channels: {info['n_channels']}", flush=True)

            save_path = save_dta_result(edf_path, signals, info, out_dir)
            processed_ok += 1
            print(f"[OK] Result saved: {save_path}", flush=True)

        except Exception as e:
            msg = f"{type(e).__name__}: {e}"
            print(f"[ERROR] Failed → {edf_path.name} | {msg}", flush=True)
            failed.append(f"{edf_path}\t{msg}")
            continue

    # Resumen final
    print("\n========== DTA SUMMARY ==========", flush=True)
    print(f"[INFO] Successfully processed: {processed_ok}", flush=True)
    print(f"[INFO] Skipped (already processed): {skipped_done}", flush=True)
    print(f"[INFO] Failed / skipped with error: {len(failed)}", flush=True)

    if failed:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fail_path = out_dir / f"failed_edfs_{ts}.txt"
        fail_path.write_text("\n".join(failed))
        print(f"[WARN] Failure list written to: {fail_path}", flush=True)


if __name__ == "__main__":
    main()
