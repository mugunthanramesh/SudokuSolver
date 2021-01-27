class SudokuSolver():
    def __init__(self, grid):
        self.grid = grid
        self.totalRows = len(grid)
        self.totalColumns = len(grid[0])

    def isSafe(self, row, col, num):
        for i in range(self.totalColumns):
            if (self.grid[row][i] == num):
                return False

        for i in range(self.totalRows):
            if (self.grid[i][col] == num):
                return False

        startRow = row - row % 3
        startCol = col - col % 3

        for i in range(3):
            for j in range(3):
                if (self.grid[i + startRow][j + startCol] == num):
                    return False

        return True

    def solveSuduko(self, row, col):

        if (row == self.totalRows - 1 and col == self.totalColumns):
            return True

        if (col == self.totalRows):
            row += 1
            col = 0

        if (self.grid[row][col] > 0):
            return self.solveSuduko(row, col + 1)

        for i in range(1, self.totalColumns + 1):
            if (self.isSafe(row, col, i)):
                self.grid[row][col] = i
                if (self.solveSuduko(row, col + 1)):
                    return True

            self.grid[row][col] = 0

        return False

    def print_matrix(self):

        for i in range(self.totalRows):
            for j in range(self.totalColumns):
                print(self.grid[i][j], end=" ")
            print("\n")