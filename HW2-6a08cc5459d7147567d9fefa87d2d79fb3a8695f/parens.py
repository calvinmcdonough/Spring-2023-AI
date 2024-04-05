class Solution(object):
    def isValid(self, s):
        """
        :type s: str
        :rtype: bool
        """
        list_of_chars = list(s)
        stack = [] # use append and pop

        if len(list_of_chars)%2 == 1:
            return False

        for char in list_of_chars:
            
            if char == '(' or char == '[' or char == '{':               
                stack.append(char)
            else:
                if char == ')' and stack.pop() != "(":
                    return False
                if char == ']' and stack.pop() != "[":
                    return False
                if char == '}' and stack.pop() != "{":
                    return False
        return True
def main():
    s1 = Solution()
    print(s1.isValid("(){}[}"))
main()
   