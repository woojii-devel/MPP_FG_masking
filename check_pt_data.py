from rdkit import Chem
from tqdm import tqdm
from multiprocessing import Pool, cpu_count


def load_smiles_txt(path):
    smiles = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) < 2:
                continue
            smiles.append(parts[1])
    return smiles


def check_one_smiles(smi):

    mol = Chem.MolFromSmiles(smi)

    if mol is None:
        return ("invalid", smi)

    if mol.GetNumAtoms() == 0:
        return ("empty", smi)

    if "." in smi:
        multi = True
    else:
        multi = False

    canonical = Chem.MolToSmiles(mol, canonical=True)

    if smi != canonical:
        return ("not_canonical", (smi, canonical))

    if multi:
        return ("multi", smi)

    return None


def check_smiles_parallel(path):

    smiles_list = load_smiles_txt(path)

    invalid = []
    empty = []
    multi_molecule = []
    not_canonical = []

    n_proc = max(1, cpu_count() - 1)

    with Pool(n_proc) as pool:

        for result in tqdm(
            pool.imap_unordered(check_one_smiles, smiles_list, chunksize=500),
            total=len(smiles_list)
        ):

            if result is None:
                continue

            tag, data = result

            if tag == "invalid":
                invalid.append(data)

            elif tag == "empty":
                empty.append(data)

            elif tag == "multi":
                multi_molecule.append(data)

            elif tag == "not_canonical":
                not_canonical.append(data)

    print(f"Total: {len(smiles_list)}")
    print(f"Invalid: {len(invalid)}")
    print(f"Empty molecule: {len(empty)}")
    print(f"Salt / multi-molecule: {len(multi_molecule)}")
    print(f"Not canonical: {len(not_canonical)}")

    return {
        "invalid": invalid,
        "empty": empty,
        "multi_molecule": multi_molecule,
        "not_canonical": not_canonical,
    }


if __name__ == "__main__":
    result = check_smiles_parallel("Vast_ai_folder/pubchem-10m.txt")