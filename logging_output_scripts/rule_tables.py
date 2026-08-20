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

def example():
    # Example Data
    sample_json = {
        "pool": [
            {
                "error_": 0.22218013605739703,
                "experience_": 180.0,
                "match": {
                    "bounds": "[[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, -0.2577443812652021], [-1.0, 1.0000000000000004], [-1.0, 1.0]]"
                },
                "is_fitted_": True,
                "model": {
                    "coef_": "[1.0514360938385392, 0.7249572682185994, -0.03290624364238745, -1.2846647087272993, -0.4346578868611826, -0.9550506556030224, -0.7675010587027079, 3.619037212605493]",
                    "intercept_": 2.911530452182757
                }
            },
            {
                "error_": 0.16363716103407536,
                "experience_": 254.0,
                "match": {
                    "bounds": "[[-1.0, 0.9675334458735618], [-1.0, 0.9475224699414337], [-0.9814730644357151, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-0.2067932069412544, 1.0], [-1.0, 1.0000000000000004], [-1.0, 1.0]]"
                },
                "is_fitted_": True,
                "model": {
                    "coef_": "[2.1690415550047475, 1.8460605905415959, 0.800345497413092, -0.03359114403043566, 0.1930557312098187, 0.7911449468606394, 0.8564932802071828, 3.4283989763275913]",
                    "intercept_": 4.802838751927556
                }
            },
            {
                "error_": 0.11399902696929883,
                "experience_": 185.0,
                "match": {
                    "bounds": "[[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-0.7996600773925618, 0.8141150573043597], [-0.849296168743414, 1.0]]"
                },
                "is_fitted_": True,
                "model": {
                    "coef_": "[1.1586769078298904, 1.1843829203565044, 0.4102647824928976, -0.4771039556090122, 0.45839173520769066, 0.055435020454600206, -0.43710119535100944, 0.2767583782755384]",
                    "intercept_": 2.2798651217744794
                }
            }
        ]
    }

    print("% Generate LaTeX tables for each rule in pool:\n")
    for idx, item in enumerate(sample_json["pool"]):
        print(
            f"% --- Tabelle {idx + 1} (Experience: {int(item['experience_'])}) ---")
        print(generate_latex_tabular(item))
        print("\n")


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
