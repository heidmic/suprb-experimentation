import json
import os
import argparse

# Static data for the middle column (Original Space) and the variable names
# This information is identical for all tables.
ORIGINAL_SPACE_DATA = [
    {"variable": "Cement [kg/m³]", "interval": [104.72, 516.78]},
    {"variable": "Blast Furnace Slag [kg/m³]", "interval": [0.0, 359.40]},
    {"variable": "Fly Ash [kg/m³]", "interval": [13.45, 200.0]},
    {"variable": "Water [kg/m³]", "interval": [122.64, 244.80]},
    {"variable": "Superplasticizer [kg/m³]", "interval": [6.02, 24.80]},
    {"variable": "Coarse Aggregate [kg/m³]", "interval": [950.16, 1145.0]},
    {"variable": "Fine Aggregate [kg/m³]", "interval": [756.14, 992.60]},
    {"variable": "Age [days]", "interval": [18.36, 365.0]}
]


def format_interval(val_range):
    """Formats an interval [min, max] cleanly for LaTeX (removes redundant trailing zeros)."""
    low = f"{val_range[0]:.2f}".rstrip('0').rstrip('.') if val_range[
                                                               0] != 0 else "0"
    high = f"{val_range[1]:.2f}".rstrip('0').rstrip('.')
    if high.endswith('.'):
        high = high[:-1]
    return f"[{low}, {high}]"


def generate_latex_tabular(pool_item, original_space=ORIGINAL_SPACE_DATA,
                           mse_orig_default=1.5310):
    """Generates the raw LaTeX tabular environment (without the floating 'table' wrapper)."""
    bounds = pool_item["match"]["bounds"]
    if isinstance(bounds, str):
        bounds = json.loads(bounds)

    coefs = pool_item["model"]["coef_"]
    if isinstance(coefs, str):
        coefs = json.loads(coefs)

    intercept = pool_item["model"]["intercept_"]
    experience = int(pool_item["experience_"])
    mse_sigma = pool_item["error_"]

    latex = []
    latex.append(r"\begin{tabular}{llll}")
    latex.append(r"\toprule")
    latex.append(
        r"& \multicolumn{1}{l}{Original Space} & \multicolumn{2}{l}{Feature Space $\sigma$} \\")
    latex.append(r"\cmidrule(r){2-2} \cmidrule(l){3-4}")
    latex.append(r"input variable & interval & interval & coef \\")
    latex.append(r"\midrule")

    for i, orig in enumerate(original_space):
        var_name = orig["variable"].replace("³", "$^3$")
        orig_interval_str = format_interval(orig["interval"])

        feat_interval = bounds[i]
        feat_interval_str = f"[{feat_interval[0]:.2f}, {feat_interval[1]:.2f}]"

        coef_val = coefs[i]
        coef_str = f"{coef_val:.2f}"

        latex.append(
            f"{var_name} & ${orig_interval_str}$ & ${feat_interval_str}$ & {coef_str} \\\\")

    latex.append(r"\midrule")

    intercept_line = r"& & \multicolumn{2}{r}{intercept$_{\sigma}$ = " + f"{intercept:.4f}" + r"} \\"
    latex.append(intercept_line)
    latex.append(r"\midrule")

    footer_text = (
            r"\multicolumn{4}{l}{"
            r"In-sample MSE$_{\text{orig}}$ " + f"{mse_orig_default:.4f}" + r" \quad "
                                                                            r"In-sample MSE$_{\sigma}$ " + f"{mse_sigma:.4f}" + r" \quad "
                                                                                                                                r"Experience " + f"{experience}"
                                                                                                                                                 r"} \\"
    )
    latex.append(footer_text)

    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}")

    return "\n".join(latex)


def wrap_in_standalone_document(tabular_code):
    """Wraps the tabular code in a compilable LaTeX standalone document."""
    doc = []
    doc.append(r"\documentclass{standalone}")
    doc.append(
        r"\usepackage{booktabs}")  # Required for \toprule, \midrule, \bottomrule, \cmidrule
    doc.append(r"\usepackage{amsmath}")  # Required for math formatting
    doc.append(r"\begin{document}")
    doc.append(tabular_code)
    doc.append(r"\end{document}")
    return "\n".join(doc)


def main():
    parser = argparse.ArgumentParser(
        description="Generates standalone, compilable LaTeX tables from a JSON file.")
    parser.add_argument("--json_file", type=str,
                        help="Path to the JSON file containing the 'pool'.")
    parser.add_argument("-o", "--output-dir", type=str, default="rule_tables",
                        help="Target directory for the generated .tex files.")

    args = parser.parse_args()

    if not os.path.exists(args.json_file):
        print(f"Error: The file '{args.json_file}' does not exist.")
        return

    try:
        with open(args.json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading JSON file: {e}")
        return

    if "pool" not in data:
        print("Error: No 'pool' key found in the JSON file.")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    for idx, item in enumerate(data["pool"]):
        experience = int(item.get("experience_", 0))
        # Filename based on index and experience
        filename = f"{args.json_file}_{idx + 1}.tex"
        filepath = os.path.join(args.output_dir, filename)

        # Generate table and wrap it in a compilable document
        table_code = generate_latex_tabular(item)
        full_document_code = wrap_in_standalone_document(table_code)

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(full_document_code)

        print(f"Successfully created: {filepath}")


if __name__ == "__main__":
    main()
