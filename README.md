# Generalizable Reasoning through Compositional Energy Minimization

<table style="width: 100%;" border="0" style="border: 0px">
    <tr>
        <td style="vertical-align: top; padding-right: 20px; width: 50%" width="50%" border="0">
            <div align="left">
                <img src="assets/.gif" width="100%">
            </div>
        </td>
        <td style="vertical-align: top; width: 50%" width="50%" border="0">
            <h3><a href="https://.github.io/"Generalizable Reasoning through Compositional Energy Minimization</a></h3>
            <p>
                <a href="http://oarga.com">Alexandru Oarga</a>*, and
                <a href="https://yilundu.com/">Yilun Du</a>*,
                <br />
                In Neural Information Processing Systems (NeurIPS), 2025
                <br />
                <a href="http://.github.io/">[Paper]</a>
                <a href="https://.github.io/">[Project Page]</a>
            </p>
        </td>
    </tr>
</table>

```
@InProceedings{oarga2025generalizable,
    author    = {Oarga, Alexandru and Du, Yilun},
    title     = {Generalizable Reasoning through Compositional Energy Minimization},
    booktitle = {Advances in Neural Information Processing Systems},
    year      = {2025}
}
```

## 1. Requirements
```
hydra-core
torch
torch-geometric
torch-scatter
torch-sparse
```

## 2. Dataset

- Data and generation scripts are available in `comp_reasoning/data/` folder.
- 3SAT data can be downloaded from [here](https://drive.google.com/file/d/1FGPS0Ox4lOp5UCaFRqwfVTy2SvtqOAuQ/view?usp=sharing)
- Crossword precomputed embeddings can be downloaded [here](https://drive.google.com/file/d/1_YteXjmiRs4T4GhBJgHCZaL7y279IRvI/view?usp=sharing)

## 3. Training

- Train N-queens
    ```bash
    ./scripts/train_nqueens.sh
    ```
- Train 3-SAT
    ```bash
    ./scripts/train_3sat.sh
    ```
- Train Graph Coloring
    ```bash
    ./scripts/train_color.sh
    ```
- Train Crosswords
    ```bash
    ./scripts/train_color.sh
    ```

## 4. Evaluation

- Evaluate N-queens
    ```bash
    ./scripts/eval_nqueens.sh
    ```
- Evaluate 3-SAT
    ```bash
    ./scripts/eval_3sat_20.sh
    ./scripts/eval_3sat_50.sh
    ```
- Evaluate Graph Coloring
    ```bash
    ./scripts/eval_color.sh
    ```
    To evaluate on multiple graph files run:
    ```bash
    ./scripts/eval_colors_dir.sh
    ```
- Evaluate Crosswords
    ```bash
    ./scripts/eval_crosswords.sh
    ```