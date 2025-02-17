list_index = [6, 7, 6, 8, 6, 9, 7, 8, 7, 9, 8, 9]

list_x = []  


for i in range(0, len(list_index), 2):
  
    if i + 1 < len(list_index):
        list_x.append([list_index[i], list_index[i + 1]])

print(list_x)

    
    