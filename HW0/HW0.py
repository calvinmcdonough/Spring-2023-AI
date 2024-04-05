def main():
	test_sign_of_number()
	test_sum_of_numbers()
	test_size_of_array()
# this function tests  to see if the number is negative positive or equall to zero
def sign_of_number(number):
	if number < 0:
		return "negative"
	if(number == 0):
		return "zero"
	if(number > 0):
		return "positive"

# this function tests the sign_of_number function the test values ar 1,0,-1 
def test_sign_of_number():
	if sign_of_number(0) == "zero":
		print("test for 0 passed")
	if sign_of_number(1) == "positive":
		print("test for positive passed")
	if sign_of_number(-1) == "negative":
		print("test for negative passed")
# this function takes in two numbers and then returns their sum
def sum_of_numbers(number1,number2):
	return number1 + number2
# this function tests the sum_of_numbers function
def test_sum_of_numbers():
	if sum_of_numbers(0,0) == 0:
		print("passed")
	if sum_of_numbers(1,1) == 2:
		print("test for positive passed")
	if sum_of_numbers(-1,1) == 0:
		print("test for negative passed")
# this function takes in an array and then returns the size of a 2 dymensional array 
# this function retrurns error if the array is less or grater than 2 dimensions
def size_of_array(array):
	if type(array[0]) == int:
		return "Error1"
	elif type(array[0][0]) != int:
		return "Error2"
	else:
		return [len(array),len(array[0])]
# this function tests the size_of_array function
def test_size_of_array():
	array = [[[0],[0]]]
	if size_of_array(array) == "Error2":
		print("passed")
	array = [[0],[0]]
	if size_of_array(array) == [2,1]:
		print("passed")
	array = [0]
	if size_of_array(array) == "Error1":
		print("passed")
main()