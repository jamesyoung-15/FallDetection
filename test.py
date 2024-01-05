import utils
import my_defs

temp_dict = {}

for parts in my_defs.IMPORTANT_PTS:
    temp_dict[parts] = {}
    temp_dict[parts]['xy'] = (0,0)
    
print(temp_dict)