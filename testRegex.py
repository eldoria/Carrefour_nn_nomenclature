import re

word = "D162"
test_alpha_num = re.search("^[a-z]+$", word.lower())
test_car_2 = re.search("^(l'|d')", word.lower())

if __name__ == "__main__":
    print(test_alpha_num)
    print(test_car_2)