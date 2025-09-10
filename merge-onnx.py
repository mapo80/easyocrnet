# merge_onnx_from_zip.py
import argparse
import os
import sys
import zipfile
import tempfile

def main():
    parser = argparse.ArgumentParser(
        description="Estrae uno zip con ONNX + external data e salva un ONNX unificato."
    )
    parser.add_argument("zip_path", help="Percorso allo zip (es. EasyOCR_Recognizer.onnx.zip)")
    parser.add_argument("-o", "--output", default=None,
                        help="Percorso file ONNX di output (default: <nomezip>_merged.onnx)")
    parser.add_argument("--model-name", default=None,
                        help="Nome del file .onnx interno da usare se nello zip ce ne sono più di uno")
    args = parser.parse_args()

    try:
        import onnx
    except ImportError:
        print("Errore: manca il pacchetto 'onnx'. Installa con: pip install onnx", file=sys.stderr)
        sys.exit(1)

    if not os.path.isfile(args.zip_path):
        print(f"File non trovato: {args.zip_path}", file=sys.stderr)
        sys.exit(1)

    if args.output is None:
        base = os.path.basename(args.zip_path)
        name, _ = os.path.splitext(base)
        # rimuove anche eventuale doppia estensione .onnx.zip
        if name.endswith(".onnx"):
            name = name[:-5]
        args.output = os.path.abspath(f"{name}_merged.onnx")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Estrai tutto
        with zipfile.ZipFile(args.zip_path, "r") as zf:
            zf.extractall(tmpdir)

        # Trova i file .onnx estratti
        onnx_files = []
        for root, _, files in os.walk(tmpdir):
            for f in files:
                if f.lower().endswith(".onnx"):
                    onnx_files.append(os.path.join(root, f))

        if not onnx_files:
            print("Nello zip non sono stati trovati file .onnx.", file=sys.stderr)
            sys.exit(1)

        if args.model_name:
            chosen = None
            for p in onnx_files:
                if os.path.basename(p) == args.model_name:
                    chosen = p
                    break
            if not chosen:
                print(f"File .onnx '{args.model_name}' non trovato nello zip.", file=sys.stderr)
                sys.exit(1)
            onnx_path = chosen
        else:
            if len(onnx_files) > 1:
                print("Trovati più file .onnx nello zip. Specifica --model-name per scegliere:", file=sys.stderr)
                for p in onnx_files:
                    rel = os.path.relpath(p, tmpdir)
                    print(f" - {rel}", file=sys.stderr)
                sys.exit(1)
            onnx_path = onnx_files[0]

        print(f"Carico il modello: {onnx_path}")
        # Carica includendo i pesi esterni
        model = onnx.load(onnx_path, load_external_data=True)

        # (opzionale) validazione
        try:
            onnx.checker.check_model(model)
        except Exception as e:
            print(f"Avviso: checker ONNX ha segnalato: {e}", file=sys.stderr)

        # Salva unificato (senza external data)
        print(f"Salvo il modello unificato in: {args.output}")
        try:
            onnx.save(model, args.output, save_as_external_data=False)
        except Exception as e:
            print("\nErrore durante il salvataggio unificato.")
            print("Probabile causa: modello > ~2GB (limite Protobuf).", file=sys.stderr)
            print("Soluzioni: quantizzare, usare float16, o mantenere external data.", file=sys.stderr)
            print(f"Dettaglio: {e}", file=sys.stderr)
            sys.exit(1)

        print("Fatto ✅")

if __name__ == "__main__":
    main()
