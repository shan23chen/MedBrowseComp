import base64
import argparse
import os
import csv



def encode_file(input_path, output_path=None):
    """Encode a file to base64 and save as .b64."""
    if output_path is None:
        output_path = input_path + ".b64"
    with open(input_path, "rb") as f:
        encoded = base64.b64encode(f.read())
    with open(output_path, "wb") as f:
        f.write(encoded)
    print(f"Encoded {input_path} -> {output_path}")


def decode_file(input_path, output_path=None):
    """Decode a base64 file back to its original format."""
    if output_path is None:
        if input_path.endswith(".b64"):
            output_path = input_path[:-4]
        else:
            output_path = input_path + ".decoded"
    with open(input_path, "rb") as f:
        decoded = base64.b64decode(f.read())
    with open(output_path, "wb") as f:
        f.write(decoded)
    print(f"Decoded {input_path} -> {output_path}")


def shift_bytes(data: bytes, shift: int) -> bytes:
    """Shift each byte in data by shift (with wraparound)."""
    return bytes((b + shift) % 256 for b in data)


def encode_shift(input_path, output_path=None, shift=3):
    """Encode a file by shifting all bytes by `shift`."""
    if output_path is None:
        output_path = input_path + f".shift{shift}"
    with open(input_path, "rb") as f:
        shifted = shift_bytes(f.read(), shift)
    with open(output_path, "wb") as f:
        f.write(shifted)
    print(f"Shift-encoded {input_path} -> {output_path} (shift={shift})")


def decode_shift(input_path, output_path=None, shift=3):
    """Decode a shift-encoded file by shifting all bytes by -shift."""
    if output_path is None:
        if input_path.endswith(f".shift{shift}"):
            output_path = input_path[:-(len(f".shift{shift}"))]
        else:
            output_path = input_path + ".decoded"
    with open(input_path, "rb") as f:
        unshifted = shift_bytes(f.read(), -shift)
    with open(output_path, "wb") as f:
        f.write(unshifted)
    print(f"Shift-decoded {input_path} -> {output_path} (shift={shift})")
    print(f"Decoded {input_path} -> {output_path}")


def encode_cell_base64(cell: str) -> str:
    return base64.b64encode(cell.encode('utf-8')).decode('utf-8')

def decode_cell_base64(cell: str) -> str:
    return base64.b64decode(cell.encode('utf-8')).decode('utf-8')

def encode_cell_shift(cell: str, shift: int) -> str:
    return ''.join(chr((ord(c) + shift) % 0x110000) for c in cell)

def decode_cell_shift(cell: str, shift: int) -> str:
    return ''.join(chr((ord(c) - shift) % 0x110000) for c in cell)

def encode_csv_cells(input_path, output_path=None, method='base64', shift=3):
    if output_path is None:
        output_path = input_path + f'.cell_{method}{shift if method=="shift" else ""}.csv'
    with open(input_path, newline='', encoding='utf-8') as infile, open(output_path, 'w', newline='', encoding='utf-8') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        for row in reader:
            if method == 'base64':
                writer.writerow([encode_cell_base64(cell) for cell in row])
            elif method == 'shift':
                writer.writerow([encode_cell_shift(cell, shift) for cell in row])
            else:
                raise ValueError(f"Unknown encoding method: {method}")
    print(f"Cell-encoded {input_path} -> {output_path} (method={method}, shift={shift if method=='shift' else ''})")

def decode_csv_cells(input_path, output_path=None, method='base64', shift=3):
    if output_path is None:
        output_path = input_path.replace('.cell_'+method+str(shift) if method=='shift' else '.cell_'+method, '')
        if output_path == input_path:
            output_path = input_path + '.decoded.csv'
    with open(input_path, newline='', encoding='utf-8') as infile, open(output_path, 'w', newline='', encoding='utf-8') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        for row in reader:
            if method == 'base64':
                writer.writerow([decode_cell_base64(cell) for cell in row])
            elif method == 'shift':
                writer.writerow([decode_cell_shift(cell, shift) for cell in row])
            else:
                raise ValueError(f"Unknown decoding method: {method}")
    print(f"Cell-decoded {input_path} -> {output_path} (method={method}, shift={shift if method=='shift' else ''})")

def encode_cell_combo(cell: str, shift: int) -> str:
    # shift, then base64 encode
    shifted = encode_cell_shift(cell, shift)
    return encode_cell_base64(shifted)

def decode_cell_combo(cell: str, shift: int) -> str:
    # base64 decode, then shift back
    base64_decoded = decode_cell_base64(cell)
    return decode_cell_shift(base64_decoded, shift)

def ensure_csv_filename(base, suffix):
    if base.endswith('.csv'):
        base_name = base[:-4]
        return f"{base_name}_{suffix}.csv"
    else:
        return f"{base}_{suffix}.csv"

def encode_csv_cells_combo(input_path, output_path=None, shift=3):
    suffix = f"cell_combo_shift{shift}_b64"
    if output_path is None:
        output_path = ensure_csv_filename(input_path, suffix)
    with open(input_path, newline='', encoding='utf-8') as infile, open(output_path, 'w', newline='', encoding='utf-8') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        for row in reader:
            writer.writerow([encode_cell_combo(cell, shift) for cell in row])
        # Ensure file ends with newline
        outfile.write('\n')
    print(f"Cell-combo-encoded {input_path} -> {output_path} (shift={shift})")

def decode_csv_cells_combo(input_path, output_path=None, shift=3):
    suffix = f"cell_combo_shift{shift}_b64"
    def strip_suffix(filename, suffix):
        if filename.endswith(f'_{suffix}.csv'):
            return filename[:-(len(f'_{suffix}.csv'))] + '.csv'
        elif filename.endswith('.csv'):
            return filename  # fallback, should not happen
        else:
            return filename + '.csv'
    if output_path is None:
        output_path = strip_suffix(input_path, suffix)
    with open(input_path, newline='', encoding='utf-8') as infile, open(output_path, 'w', newline='', encoding='utf-8') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        for row in reader:
            writer.writerow([decode_cell_combo(cell, shift) for cell in row])
        # Ensure file ends with newline
        outfile.write('\n')
    print(f"Cell-combo-decoded {input_path} -> {output_path} (shift={shift})")

def main():
    parser = argparse.ArgumentParser(description="Encode or decode CSV files using base64 or byte shift.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    enc = subparsers.add_parser("encode", help="Encode a file to base64")
    enc.add_argument("input", help="Input file path (e.g. data/final50.csv)")
    enc.add_argument("-o", "--output", help="Output file path (default: input + .b64)")

    dec = subparsers.add_parser("decode", help="Decode a base64 file")
    dec.add_argument("input", help="Input .b64 file path")
    dec.add_argument("-o", "--output", help="Output file path (default: input with .b64 removed)")

    shiftenc = subparsers.add_parser("shift-encode", help="Encode a file by shifting all bytes (default shift=3)")
    shiftenc.add_argument("input", help="Input file path")
    shiftenc.add_argument("-o", "--output", help="Output file path (default: input + .shift<shift>)")
    shiftenc.add_argument("--shift", type=int, default=3, help="Shift value (default: 3)")

    shiftdec = subparsers.add_parser("shift-decode", help="Decode a shift-encoded file (default shift=3)")
    shiftdec.add_argument("input", help="Input shift-encoded file path")
    shiftdec.add_argument("-o", "--output", help="Output file path (default: input with .shift<shift> removed)")
    shiftdec.add_argument("--shift", type=int, default=3, help="Shift value used during encoding (default: 3)")

    cellenc = subparsers.add_parser("cell-encode", help="Encode each cell in a CSV (base64 or shift)")
    cellenc.add_argument("input", help="Input CSV file path")
    cellenc.add_argument("-o", "--output", help="Output CSV file path")
    cellenc.add_argument("--method", choices=["base64", "shift"], default="base64", help="Encoding method")
    cellenc.add_argument("--shift", type=int, default=3, help="Shift value (used only if method=shift)")

    celldec = subparsers.add_parser("cell-decode", help="Decode each cell in a CSV (base64 or shift)")
    celldec.add_argument("input", help="Input encoded CSV file path")
    celldec.add_argument("-o", "--output", help="Output CSV file path")
    celldec.add_argument("--method", choices=["base64", "shift"], default="base64", help="Decoding method")
    celldec.add_argument("--shift", type=int, default=3, help="Shift value (used only if method=shift)")

    cellenccombo = subparsers.add_parser("cell-encode-combo", help="Shift then base64 encode each cell in a CSV")
    cellenccombo.add_argument("input", help="Input CSV file path")
    cellenccombo.add_argument("-o", "--output", help="Output CSV file path")
    cellenccombo.add_argument("--shift", type=int, default=3, help="Shift value (default: 3)")

    celldeccombo = subparsers.add_parser("cell-decode-combo", help="Base64 decode then shift-decode each cell in a CSV")
    celldeccombo.add_argument("input", help="Input encoded CSV file path")
    celldeccombo.add_argument("-o", "--output", help="Output CSV file path")
    celldeccombo.add_argument("--shift", type=int, default=3, help="Shift value used during encoding (default: 3)")

    args = parser.parse_args()
    if args.command == "encode":
        encode_file(args.input, args.output)
    elif args.command == "decode":
        decode_file(args.input, args.output)
    elif args.command == "shift-encode":
        encode_shift(args.input, args.output, args.shift)
    elif args.command == "shift-decode":
        decode_shift(args.input, args.output, args.shift)
    elif args.command == "cell-encode":
        encode_csv_cells(args.input, args.output, args.method, args.shift)
    elif args.command == "cell-decode":
        decode_csv_cells(args.input, args.output, args.method, args.shift)
    elif args.command == "cell-encode-combo":
        encode_csv_cells_combo(args.input, args.output, args.shift)
    elif args.command == "cell-decode-combo":
        decode_csv_cells_combo(args.input, args.output, args.shift)

if __name__ == "__main__":
    main()
