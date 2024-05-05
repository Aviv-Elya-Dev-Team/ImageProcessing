import numpy as np
from openpyxl import load_workbook


def get_cell_value(wb, cell_address):
    sheet = wb['Sheet1']
    return sheet[cell_address].value

def index_to_val(index):
    if index==0:
        return 2
    if index==1:
        return 5
    if index==2:
        return 10
    return 50

if __name__=='__main__':
    # Example usage
    file_path = "params.xlsx"  # Path to your Excel file
    wb = load_workbook(filename=file_path, read_only=True)

    cols = ['A', 'B', 'C', 'D', 'E']
    matchs = np.zeros((22, 5), dtype=np.float64)
    for i, col in enumerate(cols):
        for j in range(2, 24):
            matchs[j-2, i] = get_cell_value(wb, f'{col}{j}')
    print(matchs)
    fact_arr = [np.ones(4)]
    min = 22
    min_fact = np.zeros(4)
    for i in range(10000):
        factors = np.random.random(4)
        factors[0] = np.random.random() # .normal(np.average([a[0] for a in fact_arr]), 0.3)
        factors[1] = np.random.random() # .normal(np.average([a[1] for a in fact_arr]), 0.3)
        factors[2] = np.random.random() # .normal(np.average([a[2] for a in fact_arr]), 0.3)
        factors[3] = np.random.random() # .normal(np.average([a[3] for a in fact_arr]), 0.3)
        normed = np.copy(matchs[:, 1:])
        for j in range(len(factors)):
            normed[:, j] *= factors[j]
        miss = 0
        for j in range(normed.shape[0]):
            if index_to_val(np.argmax(normed[j, :])) != matchs[j, 0]:
                miss += 1
        if miss < min:
            min = miss
            min_fact = factors
            print(f'min is {min}!')
            fact_arr = [min_fact]
        if miss == min:
            fact_arr.append(min_fact)
    wb.close()
    print(f"hi: {min_fact}")
    print(f'2: {np.average([a[0] for a in fact_arr])}')
    print(f'5: {np.average([a[1] for a in fact_arr])}')
    print(f'10: {np.average([a[2] for a in fact_arr])}')
    print(f'50: {np.average([a[3] for a in fact_arr])}')







