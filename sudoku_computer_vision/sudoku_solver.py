# Method to print the board
def print_board(board, validate = False):
    rows = len(board)
    columns = len(board[0])
    
    if validate:
        if rows != 9 or columns != 9 :
            print("The number of rows in the board is", rows)
            print("The number of columns in the board is ", columns)
            print("The Sudoku puzzle doesnot have 81 cells. \n Please check the input")
            return False

    for i in range(rows):
        if i % 3 == 0 and i != 0 :
            print('--------------------')
        for j in range(columns):
            if j % 3 == 0 :
                print('|', end= '')
            
            if j == 8:
                print(board[i][j], end= '')
                print('|')
            else :
                print(str(board[i][j])+' ', end = "")
    return True


# Method to return the rows and columns of the cell which is empty 
def empty_cells(board):
    rows = len(board)
    columns = len(board[0])

    for i in range(rows):
        for j in range(columns):
            if board[i][j] == 0 :
                # Returns the row and column values of the cell which is empty.
                return (i, j) 
    return None


# Method to check whether its valid to place a number in the cell
def check_valid(board, num, pos):

    for i in range(len(board[0])):
        if board[pos[0]][i] == num and pos[1] != i:
            return False

    for i in range(len(board)):
        if board[i][pos[1]] == num and pos[0] != i:
            return False

    grid_x = pos[1] // 3
    grid_y = pos[0] // 3

    for i in range(grid_y*3 , grid_y*3 + 3):
        for j in range(grid_x*3, grid_x*3 + 3):

            if board[i][j] == num and pos != (i, j):
                return False

    return True

# using regression to perform backtrtacking
def sudoku_solve(board):

    empty_cell = empty_cells(board)
    if not empty_cell:
        return True
    else :
        pos = empty_cell

    for num in range(1, 10):
        valid = check_valid(board, num, pos)
        if valid:
            board[pos[0]][pos[1]] = num
            if sudoku_solve(board):
                return True
            
            board[pos[0]][pos[1]] = 0 
    return False


# status = print_board(board, validate = True)
# if status:
#     sudoku_solve(board)
#     print('\n\n Solution for sudoku is : \n')
#     print_board(board)

def solve(board):
    sudoku_solve(board)
    return board