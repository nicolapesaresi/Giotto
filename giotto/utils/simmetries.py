import numpy as np


class EquivalentBoards:
    """Specifies simmetry axes and rotations of a board game."""

    def __init__(
        self,
        rotate90: bool,
        rotate180: bool,
        rotate270: bool,
        reflect_horizontal: bool,
        reflect_vertical: bool,
        reflect_diag_nw_se: bool,
        reflect_diag_ne_sw: bool,
    ):
        self.rotate90 = rotate90
        self.rotate180 = rotate180
        self.rotate270 = rotate270
        self.reflect_horizontal = reflect_horizontal
        self.reflect_vertical = reflect_vertical
        self.reflect_diag_nw_se = reflect_diag_nw_se
        self.reflect_diag_ne_sw = reflect_diag_ne_sw

    def get_equivalent_boards(self, board: np.ndarray, policy_targets: np.ndarray | None = None):
        """Generates all unique equivalent boards applying the specified symmetries.

        Args:
            board: original board as numpy array.
            policy_targets: optionally an array with a value for each cell. In this case these are also augmented.

        Returns:
            list of equivalent boards as numpy arrays, or optionally list of board and targets pairs.
        """
        transforms = [("identity", identity)]
        if self.rotate90:
            transforms.append(("rotate90", rotate90))
        if self.rotate180:
            transforms.append(("rotate180", rotate180))
        if self.rotate270:
            transforms.append(("rotate270", rotate270))
        if self.reflect_horizontal:
            transforms.append(("reflect_horizontal", reflect_horizontal))
        if self.reflect_vertical:
            transforms.append(("reflect_vertical", reflect_vertical))
        if self.reflect_diag_nw_se:
            transforms.append(("reflect_diag_nw_se", reflect_diag_nw_se))
        if self.reflect_diag_ne_sw:
            transforms.append(("reflect_diag_ne_sw", reflect_diag_ne_sw))

        unique_boards: list[np.ndarray] = []
        results = []

        rows, cols = board.shape[:2]
        for transform_name, transform_fn in transforms:
            transformed_board = transform_fn(board)

            if any(np.array_equal(transformed_board, seen) for seen in unique_boards):
                continue
            unique_boards.append(transformed_board)

            if policy_targets is None:
                results.append(transformed_board)
            else:
                transformed_policy = self._transform_policy_targets(
                    policy_targets=policy_targets,
                    rows=rows,
                    cols=cols,
                    transform_name=transform_name,
                )
                results.append((transformed_board, transformed_policy))

        return results

    def _transform_policy_targets(
        self, policy_targets: np.ndarray, rows: int, cols: int, transform_name: str
    ) -> np.ndarray:
        """Augments policy targets accordingly to transformation."""
        policy_targets = np.asarray(policy_targets, dtype=np.float32)

        # cellwise policy (TicTacToe): length rows*cols
        if policy_targets.shape == (rows * cols,):
            policy_grid = policy_targets.reshape(rows, cols)

            if transform_name == "identity":
                transformed_grid = identity(policy_grid)
            elif transform_name == "rotate90":
                transformed_grid = rotate90(policy_grid)
            elif transform_name == "rotate180":
                transformed_grid = rotate180(policy_grid)
            elif transform_name == "rotate270":
                transformed_grid = rotate270(policy_grid)
            elif transform_name == "reflect_horizontal":
                transformed_grid = reflect_horizontal(policy_grid)
            elif transform_name == "reflect_vertical":
                transformed_grid = reflect_vertical(policy_grid)
            elif transform_name == "reflect_diag_nw_se":
                transformed_grid = reflect_diag_nw_se(policy_grid)
            elif transform_name == "reflect_diag_ne_sw":
                transformed_grid = reflect_diag_ne_sw(policy_grid)
            else:
                raise ValueError(f"Unknown transform_name: {transform_name}")

            return transformed_grid.reshape(rows * cols)

        # column policy (Connect4): length cols
        if policy_targets.shape == (cols,):
            if transform_name in ("identity", "reflect_horizontal"):
                # horizontal flip (up-down) doesn't change columns
                return policy_targets.copy()
            if transform_name == "reflect_vertical":
                # mirror left-right => reverse column probabilities
                return policy_targets[::-1].copy()

            # other transforms generally don't make sense for gravity-based Connect4.
            raise ValueError(f"Transform '{transform_name}' not supported for column-policy targets of shape {(cols,)}")

        raise ValueError(
            f"Unsupported policy_targets shape {policy_targets.shape}. Expected {(rows*cols,)} or {(cols,)} ."
        )


def identity(board):
    """Returns the board unchanged."""
    return board.copy()


def rotate90(board):  # Clockwise 90°
    """Rotates the board 90 degrees clockwise."""
    return np.rot90(board, k=-1)


def rotate180(board):
    """Rotates the board 180 degrees."""
    return np.rot90(board, k=-2)


def rotate270(board):
    """Rotates the board 270 degrees clockwise."""
    return np.rot90(board, k=-3)


def reflect_horizontal(board):
    """Flips the board upside-down."""
    return np.flipud(board)


def reflect_vertical(board):
    """Flips the board left-right."""
    return np.fliplr(board)


def reflect_diag_nw_se(board):
    """Reflects the board along the main diagonal (top-left to bottom-right)."""
    return np.transpose(board)


def reflect_diag_ne_sw(board):
    """Reflects the board along the anti-diagonal (top-right to bottom-left)."""
    return np.fliplr(np.flipud(np.transpose(board)))
