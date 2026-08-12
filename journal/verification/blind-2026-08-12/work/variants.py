from datasets import load_all


def build_variants():
    ds = load_all()
    variants = []  # list of (dataset, variant, X, XYZ, n)

    for name in ["PC10", "PC11", "FOGRA51"]:
        d = ds[name]
        k0 = d.X[:, 3] == 0
        X_cmy = d.X[k0][:, :3]
        XYZ_cmy = d.XYZ[k0]
        variants.append((name, "CMY", X_cmy, XYZ_cmy))

    d = ds["PC10"]
    variants.append(("PC10", "CMYK", d.X.copy(), d.XYZ.copy()))

    for name in ["KCMYG5", "CMYKOGV7", "CMYKOGB7"]:
        d = ds[name]
        variants.append((name, "native", d.X.copy(), d.XYZ.copy()))

    for name in ["IFRA_Age64a", "IFRA_PressJ158"]:
        d = ds[name]
        variants.append((name, "CMYK", d.X.copy(), d.XYZ.copy()))

    return variants


if __name__ == "__main__":
    for name, variant, X, XYZ in build_variants():
        print(name, variant, X.shape, XYZ.shape)
