# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "rdkit>=2024.9.1",
#     "matplotlib",
#     "numpy",
#     "pandas",
#     "altair",
#     "pillow",
#     "scikit-learn",
#     "lightgbm",
#     "scipy",
# ]
# ///

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="full", app_title="Anatomy of an ECFP")


@app.cell(hide_code=True, expand_output=True)
def _(mo):
    mo.md(r"""
    # Anatomy of an ECFP

    Bit provenance and collisions in Morgan fingerprints, and what they mean for models trained and interpreted on
    folded fingerprints.

    ECFP (Morgan) fingerprints are the default representation for QSAR, similarity search and ADMET modeling.
    Although nothing about them is learned, count ECFP with a tree ensemble is a baseline that is hard to beat in a
    meaningful way, and it regularly matches or outperforms pretrained chemical foundation models such as ChemBERTa
    and MolFormer [1–3]. The working mental model is usually vague: the fingerprint somehow encodes certain
    substructures into a bit vector. This notebook makes the encoding explicit and asks:

    > **When a model puts weight on bit 1380, which chemistry does that refer to, and how often is it several
    > unrelated environments?**

    ECFP generation in three steps:

    1. **Atom environments.** Each heavy atom starts as a radius-0 environment defined by its atom invariants
       (element, degree, H count, charge, ring membership, ...). Each iteration adds one shell of bonded neighbors,
       up to radius $r$. ECFP4 is $r = 2$.
    2. **Hashing and deduplication.** Each environment is hashed to a 32-bit identifier. Environments that cover the
       same bond set as one already seen are discarded (**redundant**).
    3. **Folding.** Identifiers are mapped onto `fpSize` bits with `bit = id mod fpSize`. Distinct identifiers on the
       same bit are a **collision**.

    The figure below shows the three steps for one molecule (originally a static figure for a paper). Molecule,
    radius, fpSize and invariants are adjustable.
    """)
    return


@app.cell(hide_code=True, expand_output=True)
def _(
    anatomy_figure,
    chiral_cb,
    env_df,
    example_dd,
    fp_name,
    fp_size,
    fpsize_dd,
    invariants_rb,
    mo,
    mol,
    radius,
    radius_sl,
    smiles,
    smiles_box,
):
    _controls = mo.vstack([
        mo.hstack([example_dd, smiles_box], widths=[1, 2], align="end"),
        mo.hstack([radius_sl, fpsize_dd, invariants_rb, chiral_cb], justify="start", gap=2, align="center"),
        mo.md(f"Current molecule: `{smiles}`"),
    ])
    if env_df is None:
        _out = mo.vstack([_controls, mo.callout(mo.md("**Could not parse that SMILES.**"), kind="danger")])
    else:
        _kept = env_df[env_df.kept]
        _n_env = len(env_df)
        _n_uid = _kept.uid.nunique()
        _n_bits = _kept.bit.nunique()
        _n_col = _kept[_kept.collision].bit.nunique()
        _stats = mo.vstack(
            [
                mo.md(f"### {fp_name} of a {mol.GetNumAtoms()}-heavy-atom molecule"),
                mo.hstack(
                    [
                        mo.stat(_n_env, label="atom environments", caption="heavy atoms x (r+1)"),
                        mo.stat(int((~env_df.kept).sum()), label="redundant", caption="same bonds as an earlier one"),
                        mo.stat(_n_uid, label="distinct features", caption="unique 32-bit identifiers"),
                        mo.stat(_n_bits, label="bits switched on", caption=f"{_n_uid - _n_bits} features lost to folding"),
                        mo.stat(_n_col, label="collided bits", caption="bits with >1 meaning",
                                direction="decrease" if _n_col else None),
                    ],
                    widths="equal",
                ),
            ]
        )
        _too_big = mol.GetNumAtoms() > 30
        _fig = anatomy_figure(mol, env_df, radius, fp_size, fp_name)
        _figure = mo.vstack(
            [
                mo.callout(
                    mo.md("Large molecule: icons get small. The bit inspector below works at any size."),
                    kind="warn",
                ) if _too_big else mo.md(""),
                _fig,
                mo.md(
                    "Each icon is one atom environment: colored disc = central atom, dark bonds = environment, grey "
                    "bonds = neighbors that enter the hash only as attachment points. Label: unfolded 32-bit identifier. "
                    "Arrow: target bit (`bit = id mod fpSize`). **Red: collision** (distinct identifiers, same bit). "
                    "Identical identifiers on different atoms (e.g. the CF3 fluorines) are one feature, not a collision."
                ),
            ]
        )
        _out = mo.vstack([_controls, _stats, _figure])
    _out
    return


@app.cell(hide_code=True, expand_output=True)
def _(
    Chem,
    Draw,
    bit_dd,
    draw_env_svg,
    env_atoms_bonds,
    env_df,
    env_smiles,
    mo,
    mol,
):
    mo.stop(bit_dd.value is None)
    _rows = env_df[env_df.kept & (env_df.bit == bit_dd.value)]
    _palette = [(1.0, 0.6, 0.6), (0.6, 0.75, 1.0), (0.6, 0.9, 0.6), (1.0, 0.85, 0.5), (0.85, 0.6, 1.0)]
    _hl_atoms, _hl_bonds, _cards = {}, {}, []
    for _k, (_uid, _g) in enumerate(_rows.groupby("uid")):
        _col = _palette[_k % len(_palette)]
        for _, _r in _g.iterrows():
            _atoms, _bonds = env_atoms_bonds(mol, int(_r.atom), int(_r.radius))
            for _a in _atoms:
                _hl_atoms.setdefault(_a, []).append(_col)
            for _b in _bonds:
                _hl_bonds.setdefault(_b, []).append(_col)
        _r0 = _g.iloc[0]
        _cards.append(
            mo.vstack(
                [
                    mo.Html(draw_env_svg(mol, int(_r0.atom), int(_r0.radius))),
                    mo.md(
                        f"<span style='display:inline-block;width:0.9em;height:0.9em;border-radius:50%;"
                        f"background:rgb({int(_col[0]*255)},{int(_col[1]*255)},{int(_col[2]*255)})'></span> "
                        f"id **{_uid}**, radius {int(_r0.radius)}, centered on "
                        f"{_r0.element} "
                        f"(atom{'s' if len(_g) > 1 else ''} {', '.join(str(a) for a in _g.atom)})  \n"
                        f"`{env_smiles(mol, int(_r0.atom), int(_r0.radius))}`"
                    ),
                ],
                align="center",
            )
        )
    _d = Draw.rdMolDraw2D.MolDraw2DSVG(420, 300)
    _d.drawOptions().useBWAtomPalette()
    _d.drawOptions().addAtomIndices = True
    _d.DrawMoleculeWithHighlights(Chem.Mol(mol), "", _hl_atoms, _hl_bonds, {}, {})
    _d.FinishDrawing()
    mo.vstack([
        mo.md(r"""
        ## Bit inspector

        All environments in the current molecule that set a given bit. If a model puts weight on that bit, these atoms
        share the attribution.
        """),
        bit_dd,
        mo.hstack(
            [
                mo.vstack([mo.md("**Atoms that share this bit**"), mo.Html(_d.GetDrawingText())]),
                mo.vstack([mo.md(f"**{len(_cards)} distinct feature(s) behind bit {bit_dd.value}**"),
                           mo.hstack(_cards, wrap=True, justify="start")]),
            ],
            widths=[1, 2],
            align="start",
        ),
    ])
    return


@app.cell(hide_code=True, expand_output=True)
def _(alt, env_df, fp_size, mo, np, pd):
    mo.stop(env_df is None)
    _uids = np.array(sorted(set(env_df[env_df.kept].uid.astype("int64"))), dtype=np.int64)
    _n = len(_uids)
    _rows = []
    for _k in range(3, 14):
        _m = 2**_k
        _obs = len(np.unique(_uids % _m))
        _rows.append(dict(fpSize=_m, kind="this molecule", lost=_n - _obs))
        _rows.append(dict(fpSize=_m, kind="random-hash expectation", lost=_n - _m * (1 - (1 - 1 / _m) ** _n)))
    _df = pd.DataFrame(_rows)
    _base = alt.Chart(_df).encode(
        x=alt.X("fpSize:Q", scale=alt.Scale(type="log", base=2), axis=alt.Axis(values=[2**k for k in range(3, 14)]),
                title="fpSize (bits, log scale)"),
        y=alt.Y("lost:Q", title="features lost to collisions"),
        color=alt.Color("kind:N", title=None, scale=alt.Scale(range=["#9a9a9a", "#e8585a"])),
    )
    _chart = (
        _base.transform_filter(alt.datum.kind == "random-hash expectation").mark_line()
        + _base.transform_filter(alt.datum.kind == "this molecule").mark_point(filled=True, size=70)
        + alt.Chart(pd.DataFrame({"fpSize": [fp_size]})).mark_rule(strokeDash=[4, 4], color="black").encode(x="fpSize:Q")
    ).properties(height=260, width="container", title=f"{_n} distinct features in this molecule")
    mo.vstack([
        mo.md(r"""
        ## Collisions vs fpSize

        Unfolded identifiers only collide on a 32-bit hash clash. For $n$ distinct identifiers hashed uniformly into $m$
        bits, the expected number of bits set is

        $$E[\text{bits on}] = m\left(1 - \left(1 - \tfrac{1}{m}\right)^{n}\right),$$

        i.e. roughly $n^2 / 2m$ features lost. Points: current molecule. Line: uniform-hash expectation.
        """),
        mo.ui.altair_chart(_chart),
    ])
    return


@app.cell(hide_code=True, expand_output=True)
def _(data, data_error, mo):
    mo.vstack([
        mo.md(r"""
        ## Collisions at dataset scale

        Training set of the **OpenADMET / ExpansionRx blind challenge** (≈5,300 compounds from lead-optimization
        campaigns, CC BY 4.0), fingerprinted with the settings above.

        * **Per molecule:** fraction of compounds with ≥1 collided bit.
        * **Per bit:** number of distinct environments mapping to each used bit across the dataset, i.e. the number of
          chemical meanings a single coefficient or SHAP value has to cover.
        """),
        mo.callout(mo.md(f"Dataset could not be loaded (`{data_error}`)."), kind="warn") if data is None else mo.md(""),
    ])
    return


@app.cell(hide_code=True, expand_output=True)
def _(alt, doc_freq, fp_name, fp_size, mo, np, pd, per_mol):
    _all_uids = np.fromiter(doc_freq.keys(), dtype=np.int64)
    _rows = []
    for _k in range(6, 15):
        _m = 2**_k
        _with_col = np.mean([len(np.unique(u % _m)) < len(u) for u in per_mol])
        _lost = np.mean([len(u) - len(np.unique(u % _m)) for u in per_mol])
        _expected = np.mean([1 - np.prod(1 - np.arange(len(u)) / _m) for u in per_mol])
        _rows.append(dict(fpSize=_m, frac=_with_col, expected=_expected, lost=_lost,
                          meanings=len(_all_uids) / len(np.unique(_all_uids % _m))))
    coll_df = pd.DataFrame(_rows)
    _x = alt.X("fpSize:Q", scale=alt.Scale(type="log", base=2), title="fpSize (bits, log scale)",
               axis=alt.Axis(values=[2**k for k in range(6, 15)]))
    _rule = alt.Chart(pd.DataFrame({"fpSize": [fp_size]})).mark_rule(strokeDash=[4, 4], color="black").encode(x="fpSize:Q")
    _long = coll_df.melt(id_vars=["fpSize"], value_vars=["frac", "expected"], var_name="kind", value_name="share")
    _long["kind"] = _long.kind.map({"frac": "observed", "expected": "random-hash expectation"})
    _c1 = (alt.Chart(_long).mark_line(point=True).encode(
        x=_x, y=alt.Y("share:Q", axis=alt.Axis(format="%"), title="molecules with ≥1 collided bit"),
        color=alt.Color("kind:N", title=None, scale=alt.Scale(range=["#e8585a", "#9a9a9a"]),
                        legend=alt.Legend(orient="top-right")),
        strokeDash=alt.StrokeDash("kind:N", legend=None, scale=alt.Scale(range=[[1, 0], [4, 3]])),
        tooltip=["fpSize", "kind", alt.Tooltip("share:Q", format=".1%")],
    ) + _rule).properties(height=240, width=380, title="Per molecule")
    _c2 = (alt.Chart(coll_df).mark_line(point=True, color="#3b6fb6").encode(
        x=_x, y=alt.Y("meanings:Q", title="distinct environments per used bit"),
        tooltip=["fpSize", alt.Tooltip("meanings:Q", format=".1f")],
    ) + _rule).properties(height=240, width=380, title="Across the dataset")
    _cur = coll_df[coll_df.fpSize == fp_size]
    _txt = (
        f"**{fp_name}**: **{_cur.frac.iloc[0]:.0%}** of molecules have ≥1 collided bit "
        f"(uniform hashing: {_cur.expected.iloc[0]:.0%}); each used bit maps to **{_cur.meanings.iloc[0]:.1f}** "
        f"distinct environments on average (**{len(_all_uids):,}** environments in total)."
        if len(_cur) else
        f"**{len(_all_uids):,}** distinct environments at these settings."
    )
    mo.vstack([mo.hstack([_c1, _c2], justify="start", gap=2), mo.md(_txt)])
    return


@app.cell(hide_code=True, expand_output=True)
def _(
    data,
    draw_env_svg,
    env_smiles,
    fp_size,
    mo,
    np,
    pd,
    per_mol,
    uid_example,
):
    from collections import Counter as _Counter

    _pairs = _Counter()
    for _u in per_mol:
        _bits = _u % fp_size
        _order = np.argsort(_bits, kind="stable")
        _b, _uu = _bits[_order], _u[_order]
        _starts = np.flatnonzero(np.r_[True, _b[1:] != _b[:-1]])
        for _s, _e in zip(_starts, np.r_[_starts[1:], len(_b)]):
            if _e - _s > 1:
                _grp = sorted(_uu[_s:_e].tolist())
                for _x in range(len(_grp)):
                    for _y in range(_x + 1, len(_grp)):
                        _pairs[(_grp[_x], _grp[_y])] += 1

    def _label(uid):
        i, a, r = uid_example[uid]
        return f"{env_smiles(data.mol[i], a, r)} (r={r})"

    pairs_df = pd.DataFrame(
        [dict(bit=a % fp_size, environment_A=_label(a), environment_B=_label(b), molecules=n)
         for (a, b), n in _pairs.most_common(25)]
    )
    _top_cards = []
    for (_a, _b), _n in _pairs.most_common(3):
        _imgs = []
        for _uid in (_a, _b):
            _i, _at, _r = uid_example[_uid]
            _imgs.append(mo.Html(draw_env_svg(data.mol[_i], _at, _r, size=(200, 160))))
        _top_cards.append(
            mo.vstack([mo.hstack(_imgs, justify="center", gap=0),
                       mo.md(f"bit **{_a % fp_size}** · both present in **{_n}** molecules")], align="center")
        )
    _pairs_out = (
        mo.vstack(
            [
                mo.hstack(_top_cards, justify="start", wrap=True, gap=2),
                mo.ui.table(pairs_df, selection=None, page_size=8, label="Top 25 colliding pairs"),
            ]
        ) if len(pairs_df) else mo.md("No collisions at these settings.")
    )
    mo.vstack([
        mo.md(r"""
        ### Most frequent colliding pairs

        At larger `fpSize` (e.g. 4096) the excess over the uniform-hash expectation comes from a few pairs of *frequent*
        environments that share a bit and therefore collide in hundreds of molecules. Below: the pairs affecting the most
        compounds at the current settings. In those molecules the two environments are indistinguishable to any model.
        """),
        _pairs_out,
    ])
    return


@app.cell(hide_code=True, expand_output=True)
def _(data, doc_freq, env_smiles, fp_size, mo, np, pd, per_mol, uid_example):
    _rows = {}
    for _uid, _df in doc_freq.items():
        _rows.setdefault(_uid % fp_size, []).append((_df, _uid))
    _on = np.zeros(fp_size, dtype=int)
    for _u in per_mol:
        _on[np.unique(_u % fp_size)] += 1
    _table = []
    for _bit, _lst in _rows.items():
        _lst.sort(reverse=True)
        _i, _a, _r = uid_example[_lst[0][1]]
        _table.append(dict(bit=_bit, meanings=len(_lst), molecules_with_bit_on=int(_on[_bit]),
                           most_common=env_smiles(data.mol[_i], _a, _r)))
    bits_table_df = pd.DataFrame(_table).sort_values(["meanings", "molecules_with_bit_on"], ascending=False).reset_index(drop=True)
    bit_members = _rows
    bits_table = mo.ui.table(bits_table_df.head(200).reset_index(drop=True), selection="single", page_size=8, label="Bits (top 200)")
    mo.vstack([
        mo.md(r"""
        ### Most overloaded bits

        Bits with the most distinct environments at the current `fpSize`. Select a row to see the environments behind it,
        ranked by document frequency. Importance assigned to such a bit is a weighted mixture of all of them.
        """),
        bits_table,
    ])
    return bit_members, bits_table, bits_table_df


@app.cell(hide_code=True, expand_output=True)
def _(
    bit_members,
    bits_table,
    bits_table_df,
    data,
    draw_env_svg,
    env_smiles,
    mo,
    uid_example,
):
    _sel = bits_table.value
    _bit = int(_sel.bit.iloc[0]) if len(_sel) else int(bits_table_df.bit.iloc[0])
    _members = sorted(bit_members[_bit], reverse=True)
    _cards = []
    for _df, _uid in _members[:12]:
        _i, _a, _r = uid_example[_uid]
        _m = data.mol[_i]
        _cards.append(
            mo.vstack(
                [
                    mo.Html(draw_env_svg(_m, _a, _r, size=(240, 190))),
                    mo.md(f"r={_r} · in **{_df}** molecules  \n`{env_smiles(_m, _a, _r)}`"),
                ],
                align="center",
            )
        )
    mo.vstack(
        [
            mo.md(f"**Bit {_bit}** collects {len(_members)} distinct environments"
                  f"{' (top 12 shown)' if len(_members) > 12 else ''}:"),
            mo.hstack(_cards, wrap=True, justify="start"),
        ]
    )
    return


@app.cell(hide_code=True, expand_output=True)
def _(
    ENDPOINTS,
    count_matrix,
    data,
    endpoint_dd,
    fit_lgbm,
    fit_ridge,
    mo,
    model_fpsize_dd,
    np,
    per_mol_info,
    r2,
):
    from scipy.stats import spearmanr as _spearmanr

    _col, _log, y_label = ENDPOINTS[endpoint_dd.value]
    _y = data[_col].to_numpy(dtype=float)
    _rows = np.flatnonzero(~np.isnan(_y))
    _y = _y[_rows]
    if _log:
        _y = np.log10(_y + 1)
    _perm = np.random.default_rng(0).permutation(len(_rows))
    _ntr = int(0.8 * len(_rows))
    train_rows, test_rows = _rows[_perm[:_ntr]], _rows[_perm[_ntr:]]
    y_train, y_test = _y[_perm[:_ntr]], _y[_perm[_ntr:]]
    m_fold = int(model_fpsize_dd.value)

    # unfolded vocabulary: ids present in >= 3 training molecules
    _df = {}
    for _i in train_rows:
        for _u in per_mol_info[_i]:
            _df[_u] = _df.get(_u, 0) + 1
    vocab = {u: j for j, u in enumerate(u for u, n in _df.items() if n >= 3)}
    train_doc_freq = _df

    def _fold_cols(ids):
        return ids % m_fold, np.ones(len(ids), bool)

    def _unf_cols(ids):
        cols = np.array([vocab.get(int(u), -1) for u in ids], dtype=np.int64)
        return cols, cols >= 0

    with mo.status.spinner("Fitting Ridge and LightGBM models ..."):
        Xf_tr = count_matrix(per_mol_info, train_rows, _fold_cols, m_fold)
        Xf_te = count_matrix(per_mol_info, test_rows, _fold_cols, m_fold)
        Xu_tr = count_matrix(per_mol_info, train_rows, _unf_cols, len(vocab))
        Xu_te = count_matrix(per_mol_info, test_rows, _unf_cols, len(vocab))
        _ridge_f, _a_f = fit_ridge(Xf_tr, y_train)
        _ridge_u, _a_u = fit_ridge(Xu_tr, y_train)
        models = {
            ("Ridge", "folded"): _ridge_f,
            ("Ridge", "unfolded"): _ridge_u,
            ("LightGBM", "folded"): fit_lgbm(Xf_tr, y_train),
            ("LightGBM", "unfolded"): fit_lgbm(Xu_tr, y_train),
        }
    test_preds = {k: m.predict(Xf_te if k[1] == "folded" else Xu_te) for k, m in models.items()}
    train_preds = {k: m.predict(Xf_tr if k[1] == "folded" else Xu_tr) for k, m in models.items()}
    # how many training molecules switch each folded bit on (for purity)
    bit_on_train = np.asarray((Xf_tr > 0).sum(axis=0)).ravel()

    model_summary = {
        k: dict(r2=r2(y_test, p), rho=_spearmanr(y_test, p)[0]) for k, p in test_preds.items()
    }

    def _stat(k):
        s = model_summary[k]
        cols = f"{m_fold} bits" if k[1] == "folded" else f"{len(vocab)} features"
        extra = f" · alpha {(_a_f if k[1] == 'folded' else _a_u):g}" if k[0] == "Ridge" else ""
        return mo.stat(f"{s['r2']:.3f}", label=f"test R², {k[0]} {k[1]} ({cols})",
                       caption=f"Spearman {s['rho']:.3f}{extra}")
    mo.vstack([
        mo.md(r"""
        ## From bits to atoms: Ridge vs LightGBM

        A benchmark of 25 pretrained embedding models on 25 datasets found "negligible or no improvement over the
        baseline ECFP" for nearly all neural models [1], and random forests on Morgan bit or count fingerprints generally
        gave the lowest error for ChEMBL Ki prediction [3]. Since these models are widely used, it is worth checking what
        they read from the fingerprint.

        Two model families, each on two featurizations:

        * **Ridge**: $\hat y = b_0 + \sum_k w_k \cdot \text{count}_k$, so the contribution of bit $k$ is
          $w_k \cdot \text{count}_k$.
        * **LightGBM**: TreeSHAP (`pred_contrib=True`) gives an additive decomposition $\hat y = \phi_0 + \sum_k \phi_k$.
          Trees can also split on the *absence* of a bit. That contribution belongs to no atom and is reported separately.
        * **Folded**: `fpSize` bits.
        * **Unfolded**: one column per identifier present in ≥3 training molecules. Collision-free, with a column count
          comparable to a 4096-bit fingerprint.

        Attribution: each column's contribution is split equally over the environment occurrences that set it, and each
        occurrence's share equally over its atoms, so the atom scores sum to the prediction minus the base value. A folded
        bit's weight is fitted on every environment that maps to it. **Purity** of an environment is the fraction of
        training molecules with that bit on that contain the environment. At purity 0.3, 70 % of the attribution is
        **borrowed** from other chemistry.
        """),
        mo.callout(mo.md(r"""
        **Scope.** This section is exploratory, not a benchmark. Live numbers come from a single random 80/20 split so that
        models refit in seconds. On one split, R² differences of a few hundredths are within split-to-split noise.
        LightGBM is untuned, and random splits of lead-optimization series are optimistic for all models. Comparative
        statements are checked with 5 × 5 repeated CV and Tukey HSD (panel below), following Ash *et al.* [4].
        """), kind="info"),
        mo.hstack([endpoint_dd, model_fpsize_dd, mo.md("*radius / invariants / chirality: from the controls at the top*")],
                  justify="start", gap=2, align="center"),
        mo.vstack(
            [
                mo.md(f"**{endpoint_dd.value}** ({y_label}): {len(train_rows)} training / {len(test_rows)} test molecules "
                      f"(random 80/20 split)."),
                mo.hstack([_stat(k) for k in models], widths="equal"),
            ]
        ),
    ])
    return (
        Xf_te,
        Xu_te,
        bit_on_train,
        m_fold,
        model_summary,
        models,
        test_preds,
        test_rows,
        train_doc_freq,
        train_preds,
        train_rows,
        vocab,
        y_label,
        y_test,
        y_train,
    )


@app.cell(hide_code=True, expand_output=True)
def _(
    alt,
    count_matrix,
    fit_lgbm,
    fit_ridge,
    m_fold,
    mo,
    model_summary,
    np,
    pd,
    per_mol_info,
    r2,
    test_rows,
    train_rows,
    y_test,
    y_train,
):
    _rows = []
    with mo.status.spinner("Scanning fingerprint lengths for both models ..."):
        for _k in range(6, 14):
            _m = 2**_k

            def _cols(ids, _m=_m):
                return ids % _m, np.ones(len(ids), bool)

            _Xtr = count_matrix(per_mol_info, train_rows, _cols, _m)
            _Xte = count_matrix(per_mol_info, test_rows, _cols, _m)
            _ridge, _ = fit_ridge(_Xtr, y_train, n_folds=3)
            _rows.append(dict(fpSize=_m, model="Ridge", r2=r2(y_test, _ridge.predict(_Xte))))
            _rows.append(dict(fpSize=_m, model="LightGBM", r2=r2(y_test, fit_lgbm(_Xtr, y_train).predict(_Xte))))
    scan_df = pd.DataFrame(_rows)
    _colors = alt.Scale(domain=["Ridge", "LightGBM"], range=["#3b6fb6", "#2a9d6f"])
    _x = alt.X("fpSize:Q", scale=alt.Scale(type="log", base=2), title="folded fpSize (bits, log scale)",
               axis=alt.Axis(values=[2**k for k in range(6, 14)]))
    _unf = pd.DataFrame([dict(model=k[0], r2=v["r2"], label=f"{k[0]} unfolded: {v['r2']:.3f}")
                         for k, v in model_summary.items() if k[1] == "unfolded"])
    _chart = (
        alt.Chart(scan_df).mark_line(point=True).encode(
            x=_x, y=alt.Y("r2:Q", title="test R²", scale=alt.Scale(zero=False)),
            color=alt.Color("model:N", scale=_colors, title=None, legend=alt.Legend(orient="bottom-right")),
            tooltip=["model", "fpSize", alt.Tooltip("r2:Q", format=".3f")])
        + alt.Chart(_unf).mark_rule(strokeDash=[6, 3]).encode(y="r2:Q", color=alt.Color("model:N", scale=_colors))
        + alt.Chart(pd.DataFrame({"fpSize": [m_fold]})).mark_rule(strokeDash=[4, 4], color="black").encode(x="fpSize:Q")
    ).properties(height=260, width=540, title="Accuracy vs. fingerprint length (dashed: unfolded)")
    _lo = scan_df[scan_df.fpSize == 128].set_index("model").r2
    _hi = scan_df[scan_df.fpSize == 8192].set_index("model").r2
    mo.hstack([_chart, mo.md(
        f"Folding costs Ridge far more than LightGBM: at 128 bits, R² {_lo['Ridge']:.2f} vs {_lo['LightGBM']:.2f}. "
        f"Trees can resolve a collided bit through interactions with other bits; Ridge has one coefficient per bit. "
        + (f"At 8192 bits Ridge overtakes LightGBM ({_hi['Ridge']:.2f} vs {_hi['LightGBM']:.2f}). "
           if _hi["Ridge"] > _hi["LightGBM"] else
           f"At 8192 bits: Ridge {_hi['Ridge']:.2f}, LightGBM {_hi['LightGBM']:.2f}. ")
        + "Ridge is usually benchmarked on folded fingerprints, where it can look much weaker than it is "
        "(compare its dashed unfolded line)."
    )], justify="start", gap=2, align="center")
    return


@app.cell(hide_code=True, expand_output=True)
def _(
    CV_METHODS,
    ENDPOINTS,
    alt,
    cv_button,
    cv_is_cached,
    cv_key,
    data,
    descriptors,
    endpoint_dd,
    m_fold,
    mo,
    np,
    pd,
    per_mol_info,
    repeated_cv,
    rm_anova_tukey,
):
    _head = mo.vstack([
        mo.md(r"""
        ### Which differences are significant?

        Protocol after Ash *et al.* [4]: 5 × 5 repeated CV (25 paired estimates per method), repeated-measures ANOVA, and
        Tukey HSD for all pairwise comparisons. The unfolded vocabulary and the Ridge α are selected on the training folds
        only. LightGBM hyperparameters are fixed.

        Intervals are mean ± HSD/2, so non-overlapping intervals ⇔ significant difference (family-wise α = 0.05). Blue:
        best. Red: significantly worse than best. Grey: not distinguishable from best. 150 fits take about a minute; the
        panel runs on demand and is cached per endpoint and setting.
        """),
        cv_button,
    ])
    mo.stop(not (cv_button.value or cv_is_cached(cv_key)), mo.vstack([_head, mo.md("*Not run yet.*")]))
    _col, _log, _ = ENDPOINTS[endpoint_dd.value]
    _y = data[_col].to_numpy(dtype=float)
    _rows = np.flatnonzero(~np.isnan(_y))
    _y = np.log10(_y[_rows] + 1) if _log else _y[_rows]
    with mo.status.progress_bar(total=25, title="5 × 5 repeated CV", remove_on_exit=True) as _bar:
        cv_df = pd.DataFrame(repeated_cv(cv_key, per_mol_info, _rows, _y, descriptors, m_fold, progress=_bar))

    _lower_better = {"MAE": True, "R2": False, "Spearman": False}
    _panels, cv_stats = [], {}
    for _metric, _low in _lower_better.items():
        _M = cv_df[cv_df.method.isin(CV_METHODS)].pivot_table(index=["rep", "fold"], columns="method",
                                                              values=_metric)[CV_METHODS].to_numpy()
        _p, _hsd = rm_anova_tukey(_M)
        _means = _M.mean(axis=0)
        _best = int(np.argmin(_means) if _low else np.argmax(_means))
        _status = ["best" if j == _best else ("significantly worse" if abs(_means[j] - _means[_best]) > _hsd
                                              else "not distinguishable") for j in range(len(CV_METHODS))]
        cv_stats[_metric] = dict(p=_p, hsd=_hsd, means=dict(zip(CV_METHODS, _means)))
        _d = pd.DataFrame(dict(method=CV_METHODS, mean=_means, lo=_means - _hsd / 2, hi=_means + _hsd / 2,
                               status=_status))
        _color = alt.Color("status:N", title=None, scale=alt.Scale(
            domain=["best", "not distinguishable", "significantly worse"], range=["#3b6fb6", "#9a9a9a", "#e8585a"]))
        _y_enc = alt.Y("method:N", sort=CV_METHODS, title=None)
        _layers = [
            alt.Chart(_d).mark_rule(strokeWidth=2).encode(x=alt.X("lo:Q", title=_metric.replace("R2", "R²"),
                                                                  scale=alt.Scale(zero=False)), x2="hi:Q",
                                                         y=_y_enc, color=_color),
            alt.Chart(_d).mark_point(filled=True, size=60).encode(
                x="mean:Q", y=_y_enc, color=_color,
                tooltip=["method", alt.Tooltip("mean:Q", format=".3f"), "status"]),
        ]
        _panels.append(alt.layer(*_layers).properties(
            width=230, height=170, title=f"{_metric.replace('R2', 'R²')}   (ANOVA p = {_p:.1e})"))
    _cv_plots = mo.vstack([alt.hconcat(*_panels).resolve_scale(color="shared"),
               mo.md(f"Null model (training mean) MAE: "
                     f"**{cv_df[cv_df.method == 'mean predictor (null)'].MAE.mean():.2f}**. On log10 endpoints, "
                     f"MAE 0.3 ≈ 2-fold error.")])
    _pairs = [
        ("ECFP unfolded · Ridge", "ECFP folded · Ridge", "unfolding, linear model"),
        ("ECFP unfolded · LightGBM", "ECFP folded · LightGBM", "unfolding, tree ensemble"),
        ("ECFP folded · LightGBM", "ECFP folded · Ridge", "tree vs linear, folded"),
        ("ECFP folded · LightGBM", "ECFP0 · LightGBM", "topology (r ≤ 2) vs atom types"),
        ("ECFP folded · LightGBM", "8 descriptors · LightGBM", "fingerprint vs 8 descriptors"),
    ]
    _rows = []
    for _a, _b, _what in _pairs:
        _row = {"comparison": f"{_a}  −  {_b}", "question": _what}
        for _metric in ["MAE", "R2", "Spearman"]:
            _s = cv_stats[_metric]
            _diff = _s["means"][_a] - _s["means"][_b]
            _sig = abs(_diff) > _s["hsd"]
            _row[_metric] = f"{_diff:+.3f} [{_diff - _s['hsd']:+.3f}, {_diff + _s['hsd']:+.3f}]{' *' if _sig else ''}"
        _rows.append(_row)
    mo.vstack([
        _head,
        _cv_plots,
        mo.vstack([
            mo.md("**Differences underlying the notebook's claims**: mean difference over 25 folds with Tukey "
                  "simultaneous 95 % interval; * = significant. ΔMAE < 0 and ΔR² > 0 favour the first method."),
            mo.ui.table(pd.DataFrame(_rows), selection=None, pagination=False),
            mo.md("Statistical significance is not practical significance [4]: with 25 paired folds, small "
                  "differences become significant, so read the interval in endpoint units. Parametric assumptions are "
                  "not checked, and there is no scaffold or time split."),
        ]),
    ])
    return


@app.cell(hide_code=True, expand_output=True)
def _(attr_model, attr_model_rb, mo, test_expl):
    _stats = [
        mo.stat(f"{test_expl.borrowed.median():.0%}", label="median borrowed attribution",
                caption="share of |atom attribution| learned from other environments"),
        mo.stat(f"{test_expl.atom_r.median():.2f}", label="median atom-level correlation",
                caption="folded vs unfolded explanation"),
        mo.stat(f"{(test_expl.sign_flips > 0).mean():.0%}", label="molecules with a sign flip",
                caption="atom with ≥25 % of max |attribution| in both maps changes sign"),
    ]
    if attr_model == "LightGBM":
        _stats.append(mo.stat(f"{test_expl.absent_share.median():.0%}", label="median 'absent bit' share",
                              caption="attribution from bits that are off (no atom)"))
    mo.vstack([
        mo.md("### Borrowed attribution"),
        attr_model_rb,
        mo.vstack([
            mo.hstack(_stats, widths="equal"),
            mo.md("Folded and unfolded maps differ even without collisions, because the model redistributes weight "
                  "over correlated features when the columns change. The *borrowed* share isolates the part due to bit "
                  "sharing. Test molecules below are sorted by borrowed share."),
        ]),
    ])
    return


@app.cell(hide_code=True, expand_output=True)
def _(mo, test_expl):
    _show = test_expl.sort_values("borrowed", ascending=False)[
        ["name", "SMILES", "measured", "pred_folded", "pred_unfolded", "borrowed", "atom_r", "sign_flips", "row"]
    ].round(3).reset_index(drop=True)
    expl_table = mo.ui.table(_show, selection="single", page_size=6, label="Test molecules",
                             format_mapping={"borrowed": "{:.0%}".format})
    expl_table
    return (expl_table,)


@app.cell(hide_code=True, expand_output=True)
def _(
    attr_model,
    attribution_svg,
    base_fold_test,
    base_unf_test,
    colorbar_html,
    contrib_fold_test,
    contrib_unf_test,
    data,
    env_smiles,
    expl_table,
    explain,
    m_fold,
    mo,
    np,
    pd,
    per_mol_info,
    set_smiles,
    test_expl,
    test_rows,
    y_label,
):
    _sel = expl_table.value
    sel_row = int(_sel.row.iloc[0]) if len(_sel) else int(test_expl.sort_values("borrowed", ascending=False).row.iloc[0])
    _k = int(np.flatnonzero(test_rows == sel_row)[0])
    sel_mol = data.mol[sel_row]
    sel_fold, sel_unf, sel_borrowed, _af, _au, sel_records = explain(
        sel_row, sel_mol, per_mol_info[sel_row], contrib_fold_test[_k], contrib_unf_test[_k])
    _smi = data.SMILES[sel_row]
    send_btn = mo.ui.button(label="Show this molecule in the anatomy figure ↑",
                            on_click=lambda _: set_smiles(_smi))
    _rec = test_expl.set_index("row").loc[sel_row]

    def _decomp(base, atoms, absent):
        txt = f"base {base:.2f} + atoms {atoms:.2f}"
        return txt + (f" + absent bits {absent:.2f}" if abs(absent) > 5e-3 else "")
    _vmax = float(max(np.abs(sel_fold).max(), np.abs(sel_unf).max(), 1e-6))
    _bmax = float(max(np.abs(sel_borrowed).max(), 1e-6))
    _panel = lambda title, scores, vmax, note: mo.vstack(
        [mo.md(f"**{title}**"), mo.Html(attribution_svg(sel_mol, scores, vmax)), mo.md(note)], align="center")
    _df = pd.DataFrame(sel_records)
    _df["environment"] = [env_smiles(sel_mol, a, r) for a, r in zip(_df.atom, _df.radius)]
    _df["sign_flip"] = (_df.contrib_folded * _df.contrib_unfolded < 0) & _df.in_vocab
    _df = _df.reindex(_df.contrib_folded.abs().sort_values(ascending=False).index)
    _n_flip = int(_df.sign_flip.sum())
    mo.vstack([
        mo.vstack([
            mo.md(f"#### {_rec['name']}  \n`{_smi}`"),
            mo.md(
                f"measured **{_rec.measured:.2f}** · {attr_model} folded **{_rec.pred_folded:.2f}** "
                f"(= {_decomp(base_fold_test[_k], sel_fold.sum(), _af)}) · "
                f"unfolded **{_rec.pred_unfolded:.2f}** (= {_decomp(base_unf_test[_k], sel_unf.sum(), _au)})"
            ),
            send_btn,
        ]),
        mo.hstack([
            mo.vstack([
                mo.hstack([
                    _panel(f"{attr_model}, folded ({m_fold} bits)", sel_fold, _vmax, "what a bit-level explanation shows"),
                    _panel(f"{attr_model}, unfolded", sel_unf, _vmax, "one column per environment, no collisions"),
                ], justify="start"),
                mo.Html(colorbar_html(_vmax, f"contribution to predicted {y_label} (orange raises, purple lowers); "
                                             "same scale in both panels")),
            ], align="center"),
            mo.vstack([
                _panel("Borrowed part of the folded map", sel_borrowed, _bmax,
                       "attribution learned from *other* environments sharing the bits"),
                mo.Html(colorbar_html(_bmax, "borrowed contribution (own scale)")),
            ], align="center"),
        ], justify="start", wrap=True, gap=2),
        mo.vstack([
            mo.md(f"**All environments in this molecule**, sorted by |folded contribution|. "
                  f"{_n_flip} contribute with the opposite sign in the unfolded model. "
                  f"Low purity: the bit's contribution mostly reflects other environments."),
            mo.ui.table(
                _df[["environment", "radius", "n", "bit", "purity", "contrib_folded", "contrib_unfolded",
                     "sign_flip"]].round(3).reset_index(drop=True),
                selection=None, page_size=10, format_mapping={"purity": "{:.0%}".format},
            ),
        ]),
    ])
    return


@app.cell(hide_code=True, expand_output=True)
def _(
    alt,
    count_matrix,
    descriptors,
    endpoint_dd,
    fit_lgbm,
    fit_ridge,
    mo,
    model_summary,
    np,
    pd,
    per_mol_info,
    r2,
    test_preds,
    test_rows,
    train_preds,
    train_rows,
    y_test,
    y_train,
):
    from sklearn.linear_model import Ridge as _Ridge2
    from scipy import sparse as _sp

    _Dtr, _Dte = descriptors[train_rows], descriptors[test_rows]
    _mu, _sd = _Dtr.mean(0), _Dtr.std(0) + 1e-9
    _Ztr, _Zte = (_Dtr - _mu) / _sd, (_Dte - _mu) / _sd
    _rows = [
        dict(model="8 descriptors · Ridge", family="descriptors", r2=r2(y_test, _Ridge2(alpha=1).fit(_Ztr, y_train).predict(_Zte))),
        dict(model="8 descriptors · LightGBM", family="descriptors",
             r2=r2(y_test, fit_lgbm(_sp.csr_matrix(_Dtr), y_train).predict(_sp.csr_matrix(_Dte).astype(np.float32)))),
    ]
    # ECFP0: unfolded radius-0 identifiers only (atom types), vocabulary from the training set
    _v0 = {}
    for _i in train_rows:
        for _u, _occ in per_mol_info[_i].items():
            if _occ[0][1] == 0 and _u not in _v0:
                _v0[_u] = len(_v0)

    def _r0_cols(ids):
        cols = np.array([_v0.get(int(u), -1) for u in ids], dtype=np.int64)
        return cols, cols >= 0

    _X0tr = count_matrix(per_mol_info, train_rows, _r0_cols, len(_v0))
    _X0te = count_matrix(per_mol_info, test_rows, _r0_cols, len(_v0))
    _rows.append(dict(model=f"ECFP0 ({len(_v0)} atom types) · Ridge", family="ECFP0",
                      r2=r2(y_test, fit_ridge(_X0tr, y_train)[0].predict(_X0te))))
    _rows.append(dict(model=f"ECFP0 ({len(_v0)} atom types) · LightGBM", family="ECFP0",
                      r2=r2(y_test, fit_lgbm(_X0tr, y_train).predict(_X0te.astype(np.float32)))))
    _proxy = []
    for _k, _s in model_summary.items():
        _rows.append(dict(model=f"ECFP {_k[1]} · {_k[0]}", family="ECFP", r2=_s["r2"]))
        _lin = _Ridge2(alpha=1).fit(_Ztr, train_preds[_k])
        _proxy.append(dict(model=f"ECFP {_k[1]} · {_k[0]}", proxy=r2(test_preds[_k], _lin.predict(_Zte))))
    perf_df = pd.DataFrame(_rows)
    proxy_df = pd.DataFrame(_proxy)
    _logp_r = np.corrcoef(_Dte[:, 0], y_test)[0, 1]

    _bars = alt.Chart(perf_df).mark_bar().encode(
        y=alt.Y("model:N", sort=None, title=None), x=alt.X("r2:Q", title="test R²"),
        color=alt.Color("family:N", scale=alt.Scale(domain=["descriptors", "ECFP0", "ECFP"], range=["#c9a227", "#9a9a9a", "#3b6fb6"]),
                        legend=None),
        tooltip=["model", alt.Tooltip("r2:Q", format=".3f")],
    ).properties(width=380, height=250, title=f"{endpoint_dd.value}: accuracy")
    _bars2 = alt.Chart(proxy_df).mark_bar(color="#8a8a8a").encode(
        y=alt.Y("model:N", sort=None, title=None),
        x=alt.X("proxy:Q", title="share of prediction variance", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])),
        tooltip=["model", alt.Tooltip("proxy:Q", format=".1%")],
    ).properties(width=380, height=130, title="ECFP predictions explained by a linear model of 8 descriptors")

    _best_desc = perf_df[perf_df.family == "descriptors"].r2.max()
    _best_ecfp = perf_df[perf_df.family == "ECFP"].r2.max()
    _best_r0 = perf_df[perf_df.family == "ECFP0"].r2.max()
    _lgbm_proxy = proxy_df.set_index("model").proxy["ECFP folded · LightGBM"]
    mo.vstack([
        mo.md(r"""
        ## Food for thought: how much of this is bulk physchem?

        Many ADMET endpoints correlate with lipophilicity, size and polarity, and feature-importance analyses often reduce
        to these. How much of the ECFP models' performance is reproduced by **eight standard descriptors** (Crippen logP,
        MW, TPSA, HBD, HBA, rotatable bonds, aromatic rings, Fsp3)?

        Related: Torrisi *et al.* found that chemical language models, regardless of size or pretraining set, mostly
        performed on par with **ECFP0** [2], i.e. atom-type counts without topology. A pretrained model that performs at
        ECFP0 level has learned little beyond composition and bulk properties.

        Checks (same split):

        1. **Descriptors:** Ridge and LightGBM on the eight descriptors.
        2. **ECFP0:** Ridge and LightGBM on unfolded radius-0 counts.
        3. **Proxy:** variance of each ECFP model's predictions explained by a linear model on the eight descriptors.
        """),
        mo.vstack([
            mo.hstack([_bars, _bars2], justify="start", gap=2),
            mo.md(
                f"**{endpoint_dd.value}**: eight descriptors reach R² **{_best_desc:.2f}** vs **{_best_ecfp:.2f}** for the "
                f"best ECFP model ({_best_desc / _best_ecfp:.0%}), and ECFP0 reaches **{_best_r0:.2f}**. Crippen logP "
                f"alone correlates with the measured values at r = {_logp_r:.2f}. A linear model on the descriptors "
                f"explains **{_lgbm_proxy:.0%}** of the variance of the folded LightGBM predictions."
            ),
            mo.md(
                "The gap to descriptors and ECFP0 is significant on every endpoint in the 5 × 5 CV, so the fingerprint "
                "models do use topology. With a random split, part of that gap is near-neighbour recall within series. "
                "Still, much of the signal is bulk physicochemistry reconstructed from substructure counts. That is "
                "worth keeping in mind before reading specific chemistry into an attribution map, or into a more "
                "complex model that only modestly beats this baseline. The descriptor share varies strongly by endpoint."
            ),
        ]),
    ])
    return


@app.cell(hide_code=True, expand_output=True)
def _(mo):
    mo.md(r"""
    ## Take-aways

    * An ECFP bit is a hash bucket. With ECFP4/2048 on ExpansionRx, 52 % of molecules have ≥1 collided bit, and each
      used bit maps to ~4 distinct environments across the dataset.
    * At 4096 bits, 43 % of molecules still have a collision (uniform hashing: 29 %). The excess comes from a few
      frequent environment pairs that share bits, so collision structure is dataset-specific.
    * Redundant environments are removed before hashing, so the feature count is below atoms × (r + 1), notably for
      small or symmetric molecules.
    * Atom attributions are exact for both Ridge (w · count) and LightGBM (TreeSHAP), but at 2048 bits part of them
      is borrowed from other environments: median 9 % (Ridge) and 13 % (LightGBM) for LogD, 10–20 % for LightGBM
      across endpoints, and 25–30 % for the worst-affected molecules. LightGBM also attributes to bits that are off
      (median 12 % for LogD, up to 46 % for the efflux ratio), which cannot be mapped onto atoms.

    The comparisons below are from 5 × 5 repeated CV with Tukey HSD [4] on all eight endpoints (ECFP4, 2048 bits) and
    can be reproduced with the panel above.

    * Ridge on unfolded counts is a much stronger model than its folded performance suggests. Unfolding improves
      Ridge significantly on LogD, KSOL, HLM and MLM CLint (LogD R² 0.901 vs 0.855, MAE 0.260 vs 0.325) and is never
      worse. Unfolded Ridge is the best of the six methods on LogD (R² 0.901 vs 0.870 for the best LightGBM) and best
      or tied-best on both protein-binding endpoints. Ridge is usually run on folded bit vectors, where it degrades
      fastest (single split, LogD, 128 bits: R² 0.54 vs 0.79 for LightGBM). Its reputation as a weak fingerprint
      baseline may partly be a folding artifact.
    * LightGBM is robust to folding: unfolding is significant only for LogD (+0.01 R²). At 2048 bits, LightGBM beats
      folded Ridge on five endpoints, is indistinguishable on LogD, and loses on both protein-binding endpoints. No
      method wins everywhere, and single-split differences of a few hundredths are noise.
    * Eight descriptors with LightGBM reach 60–76 % of the best ECFP model's CV R² (LogD 0.69 vs 0.90). A linear model
      on them explains 20–57 % of the variance of the ECFP models' predictions. ECFP0 reaches 78–91 % (LogD 0.77 vs
      0.90), consistent with chemical language models performing at ECFP0 level [2]. ECFP4 beats both significantly
      on every endpoint, but much of the apparent substructure learning is bulk physicochemistry.

    ### References

    1. M. Praski, J. Adamczyk, W. Czech. *Benchmarking Pretrained Molecular Embedding Models For Molecular
       Representation Learning.* arXiv:2508.06199 (2025). <https://arxiv.org/abs/2508.06199>
    2. M. Torrisi, S. Asadollahi, A. de la Vega de León, K. Wang, W. Copeland. *Do chemical language models provide a
       better compound representation?* NeurIPS 2023 Workshop on New Frontiers of AI for Drug Discovery and
       Development (2023). <https://doi.org/10.1101/2023.11.07.566025>
    3. J. Deng, Z. Yang, H. Wang, I. Ojima, D. Samaras, F. Wang. *A systematic study of key elements underlying
       molecular property prediction.* Nat. Commun. 14, 6395 (2023). <https://doi.org/10.1038/s41467-023-41948-6>
    4. J. R. Ash, C. Wognum, R. Rodríguez-Pérez, M. Aldeghi, A. C. Cheng, D.-A. Clevert, O. Engkvist, C. Fang,
       D. J. Price, J. M. Hughes-Oliver, W. P. Walters. *Practically Significant Method Comparison Protocols for
       Machine Learning in Small Molecule Drug Discovery.* J. Chem. Inf. Model. 65, 9398–9411 (2025).
       <https://doi.org/10.1021/acs.jcim.5c01609>

    ---
    *Data:* OpenADMET / ExpansionRx blind challenge training set (CC BY 4.0).
    *AI disclosure:* the visualization was originally hand-written for a paper. The marimo conversion and the dataset
    and modeling sections were written with help from Claude (Anthropic).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Appendix: implementation

    Helpers, data loading and model fitting. Cells execute in dataflow order, not position.
    """)
    return


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import logging
    from io import BytesIO
    from functools import lru_cache

    import numpy as np
    import pandas as pd
    import altair as alt
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.patches import ConnectionPatch, ArrowStyle
    from PIL import Image

    from rdkit import Chem, RDLogger
    from rdkit.Chem import Draw, rdDepictor
    from rdkit.Chem import rdFingerprintGenerator as rfg
    from rdkit.Chem.Draw import rdMolDraw2D

    RDLogger.DisableLog("rdApp.*")
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Liberation Sans", "Arial", "Helvetica", "DejaVu Sans"]
    return (
        ArrowStyle,
        BytesIO,
        Chem,
        ConnectionPatch,
        Draw,
        Image,
        alt,
        lru_cache,
        mo,
        np,
        pd,
        plt,
        rdDepictor,
        rdMolDraw2D,
        rfg,
    )


@app.cell(hide_code=True)
def _(mo):
    EXAMPLES = {
        "CF3-thietane": "FC(F)(F)C1CSC1",
        "Epichlorohydrin": "ClCC1OC1",
        "Paracetamol": "CC(=O)Nc1ccc(O)cc1",
        "Aspirin": "CC(=O)Oc1ccccc1C(=O)O",
        "Nicotine": "CN1CCC[C@H]1c1cccnc1",
        "Caffeine": "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
        "Ibuprofen": "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
        "Sulfamethoxazole": "Cc1cc(NS(=O)(=O)c2ccc(N)cc2)no1",
    }
    get_smiles, set_smiles = mo.state(EXAMPLES["CF3-thietane"])
    return EXAMPLES, get_smiles, set_smiles


@app.cell(hide_code=True)
def _(EXAMPLES, mo, set_smiles):
    example_dd = mo.ui.dropdown(
        options=EXAMPLES,
        value="CF3-thietane",
        label="Example molecule",
        on_change=lambda s: set_smiles(s),
    )
    smiles_box = mo.ui.text(
        placeholder="or paste any SMILES and press Enter",
        label="Custom SMILES",
        full_width=True,
        on_change=lambda s: set_smiles(s) if s.strip() else None,
    )
    radius_sl = mo.ui.slider(0, 3, value=2, step=1, label="Radius $r$", show_value=True, debounce=True)
    fpsize_dd = mo.ui.dropdown(
        options=[str(2**k) for k in range(4, 13)],
        value="128",
        label="fpSize (bits)",
    )
    invariants_rb = mo.ui.radio(
        options={"ECFP (atom invariants)": False, "FCFP (pharmacophoric features)": True},
        value="ECFP (atom invariants)",
        label="Initial invariants",
        inline=True,
    )
    chiral_cb = mo.ui.checkbox(value=False, label="include chirality")
    return (
        chiral_cb,
        example_dd,
        fpsize_dd,
        invariants_rb,
        radius_sl,
        smiles_box,
    )


@app.cell(hide_code=True)
def _(
    ArrowStyle,
    BytesIO,
    Chem,
    ConnectionPatch,
    Draw,
    Image,
    np,
    pd,
    plt,
    rdDepictor,
    rdMolDraw2D,
    rfg,
):
    # Element-wise highlight colors for the central atom of an environment (RGBA).
    CENTER_COLORS = {
        7: (0.5, 0.5, 1.0, 0.5),   # N
        8: (1.0, 0.5, 0.5, 0.5),   # O
        9: (0.5, 1.0, 0.5, 0.5),   # F
        15: (1.0, 0.7, 0.3, 0.5),  # P
        16: (1.0, 1.0, 0.5, 0.5),  # S
        17: (0.5, 1.0, 0.5, 0.5),  # Cl
        35: (0.6, 0.3, 0.2, 0.5),  # Br
        53: (0.6, 0.2, 0.6, 0.5),  # I
    }
    GREY = (0.5, 0.5, 0.5, 0.5)
    COLLISION_RGB = (1.0, 0.5, 0.5)
    ARROW = ArrowStyle.CurveFilledB(head_length=0.4, head_width=0.25)

    def make_generator(radius, fp_size=2048, features=False, chirality=False):
        """Morgan generator; the invariant generator is returned too so it stays alive."""
        inv = rfg.GetMorganFeatureAtomInvGen() if features else None
        kw = dict(radius=radius, fpSize=fp_size, includeChirality=chirality)
        if inv is not None:
            kw["atomInvariantsGenerator"] = inv
        return rfg.GetMorganGenerator(**kw), inv

    def env_atoms_bonds(mol, atom, radius):
        """Atoms and bonds covered by the environment of `atom` at `radius`."""
        if radius == 0:
            return [atom], []
        bonds = list(Chem.FindAtomEnvironmentOfRadiusN(mol, radius, atom))
        atoms = {atom}
        for b in bonds:
            bd = mol.GetBondWithIdx(b)
            atoms.update((bd.GetBeginAtomIdx(), bd.GetEndAtomIdx()))
        return sorted(atoms), bonds

    def env_smiles(mol, atom, radius):
        atoms, bonds = env_atoms_bonds(mol, atom, radius)
        if not bonds:  # radius 0: describe the atom invariants instead of a one-atom SMILES
            a = mol.GetAtomWithIdx(atom)
            sym = a.GetSymbol().lower() if a.GetIsAromatic() else a.GetSymbol()
            chg = f", charge {a.GetFormalCharge():+d}" if a.GetFormalCharge() else ""
            ring = ", ring" if a.IsInRing() else ""
            return f"{sym} (degree {a.GetDegree()}, {a.GetTotalNumHs()} H{ring}{chg})"
        return Chem.MolFragmentToSmiles(mol, atomsToUse=atoms, bondsToUse=bonds, rootedAtAtom=atom)

    def analyze(mol, radius, fp_size, features=False, chirality=False):
        """One row per (atom, radius): unfolded id, folded bit, redundancy and collision flags."""
        gen, _inv = make_generator(radius, fp_size, features, chirality)
        ao = rfg.AdditionalOutput()
        ao.AllocateBitInfoMap()
        gen.GetSparseCountFingerprint(mol, additionalOutput=ao)
        uid_of = {t: uid for uid, ts in ao.GetBitInfoMap().items() for t in ts}
        ao = rfg.AdditionalOutput()
        ao.AllocateBitInfoMap()
        gen.GetFingerprint(mol, additionalOutput=ao)
        bit_of = {t: bit for bit, ts in ao.GetBitInfoMap().items() for t in ts}

        rows = []
        for r in range(radius + 1):
            for a in mol.GetAtoms():
                key = (a.GetIdx(), r)
                rows.append(
                    dict(
                        atom=a.GetIdx(),
                        element=a.GetSymbol(),
                        radius=r,
                        kept=key in uid_of,
                        uid=uid_of.get(key),
                        bit=bit_of.get(key),
                    )
                )
        df = pd.DataFrame(rows)
        df["bit"] = df["bit"].astype("Int64")
        df["uid"] = df["uid"].astype("Int64")
        meanings = df[df.kept].groupby("bit")["uid"].nunique()
        collided = set(meanings[meanings > 1].index)
        df["collision"] = df["bit"].isin(collided)
        return df

    def draw_env_png(mol, atom, radius, size=240):
        """The paper-style environment icon: whole molecule, environment highlighted."""
        dopts = rdMolDraw2D.MolDrawOptions()
        dopts.setBackgroundColour((1, 1, 1, 0))
        dopts.useBWAtomPalette()
        center = CENTER_COLORS.get(mol.GetAtomWithIdx(atom).GetAtomicNum(), GREY)
        try:
            png = Draw.DrawMorganEnv(
                mol, atom, radius, molSize=(size, size), useSVG=False, drawOptions=dopts,
                baseRad=0.6, aromaticColor=(0.7, 0.7, 0.7, 0.0), ringColor=(0.7, 0.7, 0.7, 0.0),
                centerColor=center, extraColor=(0.2, 0.2, 0.2),
            )
        except Exception:
            return None
        return Image.open(BytesIO(png))

    def draw_env_svg(mol, atom, radius, size=(220, 180)):
        center = CENTER_COLORS.get(mol.GetAtomWithIdx(atom).GetAtomicNum(), GREY)
        dopts = rdMolDraw2D.MolDrawOptions()
        dopts.useBWAtomPalette()
        return Draw.DrawMorganEnv(
            mol, atom, radius, molSize=size, useSVG=True, drawOptions=dopts, baseRad=0.5,
            aromaticColor=(0.7, 0.7, 0.7, 0.0), ringColor=(0.7, 0.7, 0.7, 0.0),
            centerColor=center, extraColor=(0.3, 0.3, 0.3),
        )

    def anatomy_figure(mol, df, radius, fp_size, fp_name):
        """Neighborhood panels (one per radius) wired to the folded bit vector."""
        mol = Chem.Mol(mol)
        rdDepictor.SetPreferCoordGen(False)
        rdDepictor.Compute2DCoords(mol)
        xy = mol.GetConformer().GetPositions()[:, :2]
        if mol.GetNumBonds():  # normalise to unit bond length
            xy = xy / np.median([np.linalg.norm(xy[b.GetBeginAtomIdx()] - xy[b.GetEndAtomIdx()])
                                 for b in mol.GetBonds()])
        n_panels = radius + 1

        # Panel geometry in inches; equal aspect so icons sit on their atoms.
        pad = 0.75  # bond lengths of margin around the molecule
        xr = np.ptp(xy[:, 0]) + 2 * pad
        yr = np.ptp(xy[:, 1]) + 2 * pad
        pw = 5.6 if n_panels > 1 else 8.0
        scale = min(pw / xr, 6.0 / yr, 1.55)  # inches per bond length
        ph = float(max(yr * scale, 2.4))
        gap_x, gap_arrows, strip_h, bottom = 0.5, 2.6, 0.32, 0.9
        top_pad = 0.8
        fig_w = n_panels * pw + (n_panels + 1) * gap_x
        fig_h = bottom + strip_h + gap_arrows + ph + top_pad
        fig = plt.figure(figsize=(fig_w, fig_h), dpi=110)

        def frac(x_in, y_in):
            return x_in / fig_w, y_in / fig_h

        # Bit strip
        sx, sy = frac(gap_x, bottom)
        sw, sh = frac(fig_w - 2 * gap_x, strip_h)
        ax_strip = fig.add_axes([sx, sy, sw, sh])
        kept = df[df.kept]
        on_bits = set(kept.bit.astype(int))
        col_bits = set(kept[kept.collision].bit.astype(int))
        strip = np.ones((1, fp_size, 3))
        for b in on_bits:
            strip[0, b] = COLLISION_RGB if b in col_bits else (0, 0, 0)
        ax_strip.imshow(strip, aspect="auto", interpolation="nearest")
        if fp_size <= 256:
            for b in range(1, fp_size):
                ax_strip.axvline(b - 0.5, color=(0.85, 0.85, 0.85), lw=0.4)
        ax_strip.set_xticks([])
        ax_strip.set_yticks([])
        ax_strip.set_xlabel(fp_name, fontsize=20, labelpad=30)
        if col_bits:
            ax_strip.plot(sorted(col_bits), [1.0] * len(col_bits), "^", color=COLLISION_RGB, ms=7, clip_on=False)
            ax_strip.text(fp_size - 0.5, 1.4, f"▲ collision ({len(col_bits)})", fontsize=11,
                          color=COLLISION_RGB, ha="right", va="top")

        icon_in = float(min(0.8 * scale, 1.2))
        label_fs = float(np.clip(icon_in * 7.5, 5, 9))
        y0_in = bottom + strip_h + gap_arrows
        for r in range(n_panels):
            px_in = gap_x + r * (pw + gap_x)
            ax = fig.add_axes([*frac(px_in, y0_in), *frac(pw, ph)])
            cx, cy = (xy[:, 0].max() + xy[:, 0].min()) / 2, (xy[:, 1].max() + xy[:, 1].min()) / 2
            ax.set_xlim(cx - pw / (2 * scale), cx + pw / (2 * scale))
            ax.set_ylim(cy - ph / (2 * scale), cy + ph / (2 * scale))
            ax.axis("off")
            ax.set_title(f"Atomic neighborhoods at radius {r}", fontsize=17, pad=14)
            for bd in mol.GetBonds():
                i, j = bd.GetBeginAtomIdx(), bd.GetEndAtomIdx()
                ax.plot(xy[[i, j], 0], xy[[i, j], 1], ls=":", lw=4, color="blue", alpha=0.4)

            sub = df[df.radius == r].set_index("atom")
            for a in range(mol.GetNumAtoms()):
                ix_in = px_in + (xy[a, 0] - (cx - pw / (2 * scale))) * scale
                iy_in = y0_in + (xy[a, 1] - (cy - ph / (2 * scale))) * scale
                icon_ax = fig.add_axes([*frac(ix_in - icon_in / 2, iy_in - icon_in / 2), *frac(icon_in, icon_in)])
                img = draw_env_png(mol, a, r)
                if img is not None:
                    icon_ax.imshow(img)
                icon_ax.set_xticks([])
                icon_ax.set_yticks([])
                for sp in icon_ax.spines.values():
                    sp.set_color((0.4, 0.4, 0.4))
                    sp.set_linewidth(0.6)
                row = sub.loc[a]
                if not row.kept:
                    icon_ax.text(0.5, -0.06, "REDUNDANT", transform=icon_ax.transAxes, fontsize=label_fs,
                                 color=COLLISION_RGB, ha="center", va="top")
                    for sp in icon_ax.spines.values():
                        sp.set_linestyle("--")
                    continue
                bit = int(row.bit)
                icon_ax.text(0.5, -0.06, f"{int(row.uid)}", transform=icon_ax.transAxes,
                             fontsize=label_fs, ha="center", va="top")
                bit_x = sx + (bit + 0.5) / fp_size * sw
                icon_x = (ix_in) / fig_w
                con = ConnectionPatch(
                    xyA=(0.5, 0.0), coordsA="axes fraction", axesA=icon_ax,
                    xyB=(bit, -0.5), coordsB="data", axesB=ax_strip,
                    connectionstyle=f"angle,angleA=-90,angleB={'-' if bit_x > icon_x else ''}10,rad=12",
                    color=COLLISION_RGB if row.collision else "grey",
                    lw=1.6, arrowstyle=ARROW, alpha=0.9,
                )
                ax_strip.add_artist(con)
        return fig

    return (
        analyze,
        anatomy_figure,
        draw_env_svg,
        env_atoms_bonds,
        env_smiles,
        make_generator,
    )


@app.cell(hide_code=True)
def _(Chem, get_smiles):
    smiles = get_smiles().strip()
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    return mol, smiles


@app.cell(hide_code=True)
def _(analyze, chiral_cb, fpsize_dd, invariants_rb, mol, radius_sl):
    radius = radius_sl.value
    fp_size = int(fpsize_dd.value)
    use_features = invariants_rb.value
    chirality = chiral_cb.value
    fp_name = f"{'FCFP' if use_features else 'ECFP'}{2 * radius}_{fp_size}"
    env_df = analyze(mol, radius, fp_size, use_features, chirality) if mol is not None else None
    return chirality, env_df, fp_name, fp_size, radius, use_features


@app.cell(hide_code=True)
def _(env_df, mo):
    mo.stop(env_df is None)
    _kept = env_df[env_df.kept]
    _opts = {}
    for _bit, _g in _kept.groupby("bit"):
        _n = _g.uid.nunique()
        _opts[f"bit {_bit}: {_n} feature{'s' if _n > 1 else ''}{'  (collision)' if _n > 1 else ''}"] = int(_bit)
    _first_col = next((k for k, v in _opts.items() if "collision" in k), next(iter(_opts), None))
    bit_dd = mo.ui.dropdown(options=_opts, value=_first_col, label="Bit")
    return (bit_dd,)


@app.cell(hide_code=True)
def _(Chem, mo, pd):
    DATA_URL = (
        "https://huggingface.co/datasets/openadmet/openadmet-expansionrx-challenge-train-data/"
        "resolve/main/expansion_data_train.csv"
    )
    try:
        with mo.status.spinner("Downloading ExpansionRx training data ..."):
            data = pd.read_csv(DATA_URL)
        data["mol"] = [Chem.MolFromSmiles(s) for s in data.SMILES]
        data = data[data.mol.notna()].reset_index(drop=True)
        data_error = None
    except Exception as e:  # offline: the rest of the notebook still works
        data, data_error = None, str(e)
    return data, data_error


@app.cell(hide_code=True)
def _(lru_cache, make_generator, np, rfg):
    def dataset_features(mols, radius, features, chirality):
        """Unfolded ids per molecule, document frequency per id, one example (mol, atom, r) per id,
        and the full bit-info map {id: ((atom, r), ...)} per molecule."""
        gen, _inv = make_generator(radius, 2048, features, chirality)
        per_mol, doc_freq, example, info = [], {}, {}, []
        for i, m in enumerate(mols):
            ao = rfg.AdditionalOutput()
            ao.AllocateBitInfoMap()
            gen.GetSparseCountFingerprint(m, additionalOutput=ao)
            bim = ao.GetBitInfoMap()
            per_mol.append(np.fromiter(bim.keys(), dtype=np.int64, count=len(bim)))
            info.append(dict(bim))
            for uid, ts in bim.items():
                doc_freq[uid] = doc_freq.get(uid, 0) + 1
                if uid not in example:
                    example[uid] = (i, *ts[0])
        return per_mol, doc_freq, example, info

    @lru_cache(maxsize=16)
    def dataset_features_cached(key, radius, features, chirality):
        return dataset_features(_MOL_REGISTRY[key], radius, features, chirality)

    _MOL_REGISTRY = {}

    def register_mols(key, mols):
        _MOL_REGISTRY[key] = mols
        return key

    return dataset_features_cached, register_mols


@app.cell(hide_code=True)
def _(
    chirality,
    data,
    data_error,
    dataset_features_cached,
    mo,
    radius,
    register_mols,
    use_features,
):
    mo.stop(data is None, mo.callout(mo.md(f"Dataset could not be loaded (`{data_error}`)."), kind="warn"))
    _key = register_mols("expansionrx", list(data.mol))
    with mo.status.spinner("Fingerprinting the dataset ..."):
        per_mol, doc_freq, uid_example, per_mol_info = dataset_features_cached(_key, radius, use_features, chirality)
    return doc_freq, per_mol, per_mol_info, uid_example


@app.cell(hide_code=True)
def _(mo):
    ENDPOINTS = {
        "LogD": ("LogD", False, "LogD"),
        "Kinetic solubility": ("KSOL", True, "log10(KSOL + 1), µM"),
        "HLM CLint": ("HLM CLint", True, "log10(HLM CLint + 1), mL/min/kg"),
        "MLM CLint": ("MLM CLint", True, "log10(MLM CLint + 1), mL/min/kg"),
        "Caco-2 Papp A>B": ("Caco-2 Permeability Papp A>B", True, "log10(Papp + 1), 1e-6 cm/s"),
        "Caco-2 efflux ratio": ("Caco-2 Permeability Efflux", True, "log10(ER + 1)"),
        "Mouse plasma protein binding": ("MPPB", True, "log10(% unbound + 1)"),
        "Mouse brain protein binding": ("MBPB", True, "log10(% unbound + 1)"),
    }
    endpoint_dd = mo.ui.dropdown(options=list(ENDPOINTS), value="LogD", label="Endpoint")
    model_fpsize_dd = mo.ui.dropdown(
        options=[str(2**k) for k in range(6, 14)], value="2048", label="Folded model fpSize"
    )
    return ENDPOINTS, endpoint_dd, model_fpsize_dd


@app.cell(hide_code=True)
def _(np):
    from scipy import sparse as _sparse
    from sklearn.linear_model import Ridge as _Ridge
    import lightgbm as _lgb

    LGBM_PARAMS = dict(n_estimators=200, learning_rate=0.1, num_leaves=31, min_child_samples=10,
                       subsample=0.8, subsample_freq=1, colsample_bytree=0.5, random_state=0, verbose=-1)

    def count_matrix(infos, rows, col_fn, n_cols):
        """Sparse count matrix; col_fn maps an id-array to (column-array, keep-mask)."""
        r_idx, c_idx, vals = [], [], []
        for r, i in enumerate(rows):
            ids = np.fromiter(infos[i].keys(), dtype=np.int64, count=len(infos[i]))
            cnt = np.array([len(v) for v in infos[i].values()], dtype=float)
            cols, keep = col_fn(ids)
            r_idx.append(np.full(keep.sum(), r))
            c_idx.append(cols[keep])
            vals.append(cnt[keep])
        return _sparse.csr_matrix(
            (np.concatenate(vals), (np.concatenate(r_idx), np.concatenate(c_idx))), shape=(len(rows), n_cols)
        )

    def fit_ridge(X, y, alphas=(0.3, 1, 3, 10, 30, 100), n_folds=4, seed=0):
        """Pick alpha by k-fold CV on the training set, refit on all of it."""
        folds = np.random.default_rng(seed).integers(0, n_folds, X.shape[0])
        cv = []
        for a in alphas:
            sse = 0.0
            for f in range(n_folds):
                tr, va = folds != f, folds == f
                m = _Ridge(alpha=a).fit(X[tr], y[tr])
                sse += ((m.predict(X[va]) - y[va]) ** 2).sum()
            cv.append(sse)
        best = alphas[int(np.argmin(cv))]
        return _Ridge(alpha=best).fit(X, y), best

    def fit_lgbm(X, y):
        """Fixed, sensible hyperparameters: this is a baseline, not a tuned model."""
        return _lgb.LGBMRegressor(**LGBM_PARAMS).fit(X.astype(np.float32), y)

    def contributions(model, X):
        """Per-column contribution to each prediction, and the base value.

        Ridge: w_k * x_k (exact). LightGBM: TreeSHAP values (exact additive decomposition)."""
        if hasattr(model, "coef_"):
            return np.asarray(X.multiply(model.coef_).todense()), np.full(X.shape[0], model.intercept_)
        phi = model.predict(X.astype(np.float32), pred_contrib=True)
        phi = phi.toarray() if hasattr(phi, "toarray") else np.asarray(phi)
        phi = phi.astype(np.float64)
        return phi[:, :-1], phi[:, -1]

    def r2(y, p):
        return 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()

    return contributions, count_matrix, fit_lgbm, fit_ridge, r2


@app.cell(hide_code=True)
def _(fit_lgbm, fit_ridge, np):
    from scipy import sparse as _sp3
    from scipy.stats import f as _fdist, spearmanr as _spr, studentized_range as _sr

    _CV_CACHE = {}

    CV_METHODS = ["8 descriptors · LightGBM", "ECFP0 · LightGBM", "ECFP folded · Ridge", "ECFP unfolded · Ridge",
                  "ECFP folded · LightGBM", "ECFP unfolded · LightGBM"]

    def repeated_cv(key, infos, rows, y, desc, m_fold, n_rep=5, n_folds=5, progress=None):
        """5x5 repeated CV of all methods on the same folds; one row per (repeat, fold, method)."""
        if key in _CV_CACHE:
            return _CV_CACHE[key]
        # one matrix over every identifier, from which all featurizations are column subsets
        index, is_r0, R, C, V = {}, [], [], [], []
        for r, i in enumerate(rows):
            for u, occ in infos[i].items():
                j = index.setdefault(u, len(index))
                if j == len(is_r0):
                    is_r0.append(occ[0][1] == 0)
                R.append(r)
                C.append(j)
                V.append(len(occ))
        X_all = _sp3.csr_matrix((V, (R, C)), shape=(len(rows), len(index)), dtype=np.float64)
        is_r0 = np.array(is_r0)
        uids = np.fromiter(index.keys(), dtype=np.int64, count=len(index))
        fold_map = _sp3.csr_matrix((np.ones(len(uids)), (np.arange(len(uids)), uids % m_fold)),
                                   shape=(len(uids), m_fold))
        X_fold = (X_all @ fold_map).tocsr()
        D = desc[rows]
        out = []
        for rep in range(n_rep):
            parts = np.array_split(np.random.default_rng(100 + rep).permutation(len(rows)), n_folds)
            for f in range(n_folds):
                te = parts[f]
                tr = np.concatenate([parts[g] for g in range(n_folds) if g != f])
                doc = np.asarray((X_all[tr] > 0).sum(axis=0)).ravel()
                X_unf = X_all[:, np.flatnonzero(doc >= 3)].tocsr()
                X_r0 = X_all[:, np.flatnonzero(is_r0 & (doc >= 1))].tocsr()
                ytr, yte = y[tr], y[te]
                preds = {
                    "8 descriptors · LightGBM": fit_lgbm(_sp3.csr_matrix(D[tr]), ytr).predict(
                        _sp3.csr_matrix(D[te]).astype(np.float32)),
                    "ECFP0 · LightGBM": fit_lgbm(X_r0[tr], ytr).predict(X_r0[te].astype(np.float32)),
                    "ECFP folded · Ridge": fit_ridge(X_fold[tr], ytr, alphas=(1, 3, 10, 30), n_folds=3)[0].predict(X_fold[te]),
                    "ECFP unfolded · Ridge": fit_ridge(X_unf[tr], ytr, alphas=(1, 3, 10, 30), n_folds=3)[0].predict(X_unf[te]),
                    "ECFP folded · LightGBM": fit_lgbm(X_fold[tr], ytr).predict(X_fold[te].astype(np.float32)),
                    "ECFP unfolded · LightGBM": fit_lgbm(X_unf[tr], ytr).predict(X_unf[te].astype(np.float32)),
                    "mean predictor (null)": np.full(len(te), ytr.mean()),
                }
                for meth, p in preds.items():
                    out.append(dict(rep=rep, fold=f, method=meth, MAE=np.abs(p - yte).mean(),
                                    R2=1 - ((p - yte) ** 2).sum() / ((yte - yte.mean()) ** 2).sum(),
                                    Spearman=_spr(p, yte)[0] if np.ptp(p) > 0 else 0.0))
                if progress is not None:
                    progress.update()
        _CV_CACHE[key] = out
        return out

    def rm_anova_tukey(M, alpha=0.05):
        """Repeated-measures ANOVA (subjects = CV folds) and the Tukey HSD for an n x k score matrix."""
        n, k = M.shape
        grand = M.mean()
        ss_m = n * ((M.mean(axis=0) - grand) ** 2).sum()
        ss_s = k * ((M.mean(axis=1) - grand) ** 2).sum()
        ss_e = ((M - grand) ** 2).sum() - ss_m - ss_s
        df_m, df_e = k - 1, (k - 1) * (n - 1)
        p = _fdist.sf((ss_m / df_m) / (ss_e / df_e), df_m, df_e)
        hsd = _sr.ppf(1 - alpha, k, df_e) * np.sqrt(ss_e / df_e / n)
        return p, hsd

    def cv_is_cached(key):
        return key in _CV_CACHE

    return CV_METHODS, cv_is_cached, repeated_cv, rm_anova_tukey


@app.cell(hide_code=True)
def _(chirality, endpoint_dd, m_fold, mo, radius, use_features):
    cv_key = (endpoint_dd.value, radius, use_features, chirality, m_fold)
    cv_button = mo.ui.run_button(label=f"Run 5 × 5 repeated CV for {endpoint_dd.value} (~1 min)")
    return cv_button, cv_key


@app.cell(hide_code=True)
def _(bit_on_train, env_atoms_bonds, m_fold, np, train_doc_freq, vocab):
    _env_cache = {}

    def explain(mol_key, mol, info, c_fold, c_unf):
        """Map per-column contributions of one molecule (folded and unfolded model) onto atoms.

        Returns folded, unfolded and borrowed atom scores, the contribution of absent columns for both
        models, and a per-environment record list."""
        n = mol.GetNumAtoms()
        fold, unf, borrowed = np.zeros(n), np.zeros(n), np.zeros(n)
        bit_count = {}
        for uid, occ in info.items():
            bit_count[uid % m_fold] = bit_count.get(uid % m_fold, 0) + len(occ)
        records, used_unf = [], set()
        for uid, occ in info.items():
            bit = uid % m_fold
            share_f = c_fold[bit] / bit_count[bit]  # per occurrence
            j = vocab.get(uid)
            share_u = c_unf[j] / len(occ) if j is not None else 0.0
            if j is not None:
                used_unf.add(j)
            purity = min(1.0, train_doc_freq.get(uid, 0) / bit_on_train[bit]) if bit_on_train[bit] else 0.0
            for a, r in occ:
                key = (mol_key, a, r)
                if key not in _env_cache:
                    _env_cache[key] = env_atoms_bonds(mol, a, r)[0]
                atoms = _env_cache[key]
                fold[atoms] += share_f / len(atoms)
                unf[atoms] += share_u / len(atoms)
                borrowed[atoms] += share_f * (1 - purity) / len(atoms)
            records.append(dict(uid=uid, atom=occ[0][0], radius=occ[0][1], n=len(occ), bit=bit, purity=purity,
                                contrib_folded=share_f * len(occ), contrib_unfolded=share_u * len(occ),
                                in_vocab=j is not None))
        absent_f = float(c_fold.sum() - sum(c_fold[b] for b in bit_count))
        absent_u = float(c_unf.sum() - sum(c_unf[j] for j in used_unf))
        return fold, unf, borrowed, absent_f, absent_u, records

    return (explain,)


@app.cell(hide_code=True)
def _(mo):
    attr_model_rb = mo.ui.radio(options=["Ridge", "LightGBM"], value="Ridge", label="Attribution for",
                                inline=True)
    return (attr_model_rb,)


@app.cell(hide_code=True)
def _(
    Xf_te,
    Xu_te,
    attr_model_rb,
    contributions,
    data,
    explain,
    mo,
    models,
    np,
    pd,
    per_mol_info,
    test_preds,
    test_rows,
    y_test,
):
    attr_model = attr_model_rb.value
    with mo.status.spinner(f"Attributing every test molecule ({attr_model}) ..."):
        contrib_fold_test, base_fold_test = contributions(models[(attr_model, "folded")], Xf_te)
        contrib_unf_test, base_unf_test = contributions(models[(attr_model, "unfolded")], Xu_te)
        _out = []
        for _k, _i in enumerate(test_rows):
            _f, _u, _b, _af, _au, _ = explain(int(_i), data.mol[_i], per_mol_info[_i],
                                              contrib_fold_test[_k], contrib_unf_test[_k])
            _abs = np.abs(_f).sum()
            _big = (np.abs(_f) > 0.25 * np.abs(_f).max()) & (np.abs(_u) > 0.25 * np.abs(_u).max())
            _out.append(dict(
                row=int(_i), name=data["Molecule Name"][_i], SMILES=data.SMILES[_i], measured=y_test[_k],
                pred_folded=test_preds[(attr_model, "folded")][_k],
                pred_unfolded=test_preds[(attr_model, "unfolded")][_k],
                borrowed=np.abs(_b).sum() / _abs if _abs else 0.0,
                atom_r=np.corrcoef(_f, _u)[0, 1] if np.std(_f) > 0 and np.std(_u) > 0 else np.nan,
                sign_flips=int((np.sign(_f) != np.sign(_u))[_big].sum()),
                absent_share=abs(_af) / (abs(_af) + _abs) if (abs(_af) + _abs) else 0.0,
            ))
    test_expl = pd.DataFrame(_out)
    return (
        attr_model,
        base_fold_test,
        base_unf_test,
        contrib_fold_test,
        contrib_unf_test,
        test_expl,
    )


@app.cell(hide_code=True)
def _(Chem, Draw, np, plt):
    from matplotlib import colors as _mcolors

    _CMAP = plt.get_cmap("PuOr_r")

    def attribution_svg(mol, scores, vmax, size=(380, 300)):
        """Atom-highlight map with a shared, symmetric color scale."""
        # no extra chiral Hs: atom indices must stay aligned with `scores`
        mol = Draw.rdMolDraw2D.PrepareMolForDrawing(Chem.Mol(mol), addChiralHs=False)
        norm = _mcolors.Normalize(-vmax, vmax)
        cols, radii = {}, {}
        for a, s in enumerate(scores):
            cols[a] = tuple(_CMAP(norm(s))[:3])
            radii[a] = 0.25 + 0.35 * min(1.0, abs(s) / vmax) if vmax else 0.25
        bonds, bcols = [], {}
        for b in mol.GetBonds():
            s = (scores[b.GetBeginAtomIdx()] + scores[b.GetEndAtomIdx()]) / 2
            bonds.append(b.GetIdx())
            bcols[b.GetIdx()] = tuple(_CMAP(norm(s))[:3])
        d = Draw.rdMolDraw2D.MolDraw2DSVG(*size)
        o = d.drawOptions()
        o.useBWAtomPalette()
        o.fillHighlights = True
        o.atomHighlightsAreCircles = True
        o.highlightBondWidthMultiplier = 12
        d.DrawMolecule(mol, highlightAtoms=list(cols), highlightAtomColors=cols, highlightAtomRadii=radii,
                       highlightBonds=bonds, highlightBondColors=bcols)
        d.FinishDrawing()
        return d.GetDrawingText()

    def colorbar_html(vmax, label):
        stops = ", ".join(_mcolors.to_hex(_CMAP(x)) for x in np.linspace(0, 1, 9))
        return (f"<div style='font-size:0.8rem'><div style='height:10px;width:260px;border-radius:3px;"
                f"background:linear-gradient(to right,{stops})'></div><div style='display:flex;"
                f"justify-content:space-between;width:260px'><span>−{vmax:.2f}</span><span>0</span>"
                f"<span>+{vmax:.2f}</span></div><div>{label}</div></div>")

    return attribution_svg, colorbar_html


@app.cell(hide_code=True)
def _(data, np):
    from rdkit.Chem import Crippen as _Crippen, Descriptors as _Descriptors, rdMolDescriptors as _rmd

    DESCRIPTOR_NAMES = ["logP (Crippen)", "MW", "TPSA", "HBD", "HBA", "rotatable bonds", "aromatic rings", "Fsp3"]

    def _desc(m):
        return [_Crippen.MolLogP(m), _Descriptors.MolWt(m), _rmd.CalcTPSA(m), _rmd.CalcNumHBD(m),
                _rmd.CalcNumHBA(m), _rmd.CalcNumRotatableBonds(m), _rmd.CalcNumAromaticRings(m),
                _rmd.CalcFractionCSP3(m)]

    descriptors = np.array([_desc(m) for m in data.mol])
    return (descriptors,)


if __name__ == "__main__":
    app.run()
