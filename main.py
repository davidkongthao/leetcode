from math import e, log

values = {
    'I', 1,
    'V': 5,
    'X': 10,
    'L': 50,
    'C': 100,
    'D': 500,
    'M': 1000,
    'IV': 4,
    'IX': 9,
    'XL': 40,
    'XC': 90,
    'CD': 400,
    'CM': 900
}

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:

    def twoSum(self, nums: list[int], target: int) -> list[int]:
        '''
        Answer to Two Sum (Easy) question on LeetCode.
        
        https://leetcode.com/problems/two-sum
        '''
        w = len(nums)
        for i in range(w):
            for x in range(w):
                if nums[i] + nums[x] == target and i != x:
                    return [i, x]
    
    def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
        '''
        Answer to Add Two Numbers (Medium) question on LeetCode.

        https://leetcode.com/problems/add-two-numbers
        '''
        # Head of the new linked list - this is the head of the resultant list
        head = None
        # Reference of head which is null at this point
        temp = None
        # Carry 
        carry = None
        # Loop for the two lists
        while l1 is not None or l2 is not None:
            # At the start of each iteration, we should add carry from the last iteration
            sum_value = carry
            # Since the lengths of the lists may be unequal, we are checking if the current node is null for one of the lists
            if l1 is not None:
                sum_value += l1.val
                l1 = l1.next
            if l2 is not None:
                sum_value += l2.val
                l2 = l2.next

            node = ListNode(sum_value % 10)

            carry = sum_value // 10

            if temp is None:
                temp = head = node
            else:
                temp.next = node
                temp = temp.next

        print(carry)
        if carry > 0:
            temp.next = ListNode(carry)
        return head

    
    def isPalindrome(self, x: int) -> bool:
        """
        Answer to Palindrome Number (Easy) in LeetCode.

        https://leetcode.com/problems/palindrome-addTwoNumbers

        """
        return str(x) == str(x)[::-1]
    
    def isValidPalindrome(self, s: str) -> bool:
        _s = ''.join(x.lower() for x in s if x.isalnum())
        return _s == _s[::-1]
    
    def isValid(self, s: str) -> bool:
        """
        Answer to Valid Parentheses (Easy) in LeetCode.

        https://leetcode.com/problems/valid-parentheses
        """
        d = {'(':')', '{':'}', '[':']'}
        stack = []
        for i in s:
            if i in d:
                stack.append(i)
            elif len(stack) == 0 or d[stack.pop()] != i:
                return False
        return len(stack) == 0

    def isRoman(self, s: str) -> int:
        """
        Answer to Roman to Integer

        https://leetcode.com/problems/roman-to-integer
        """
        roman = {
                'I': 1,
                'V': 5,
                'X': 10,
                'L': 50,
                'C': 100,
                'D': 500,
                'M': 1000
        }

        num = 0
        s = s.replace('IV', 'IIII').replace('IX', 'VIIII')
        s = s.replace('XL', 'XXXX').replace('XC', 'LXXXX')
        s = s.replace('CD', 'CCCC').replace('CM', 'DCCCC')
        for char in s:
            num += roman[char]

        return num
    
    def isRomanStandard(self, s: str) -> int:
        """
        Conventional Answer to Roman to Integer
    
        https://leetcode.com/problems/roman-to-integer

        This takes a left-to-right approach
        """

        total = 0
        i = 0 
        while i < len(s):
            if i < len(s) -1 and s[i:i+2] in values:
                total += values[s[i:i+2]]
                i += 2
            else:
                total += values[s[i]]
                i += 1
        return total
    
    def longestCommonPrefix(self, strs: list[str]) -> str:
        """
        Solution to Longest Common Prefix (Easy)

        https://leetcode.com/problems/longest-common-prefix

        """
        if not strs:
            return ""
        shortest = min(strs, key=len)
        for i, ch in enumerate(shortest):
            for other in strs:
                if other[i] != ch:
                    return shortest[:i]
        return shortest

    def mergeTwoLists(self, list1, list2):
        if list1 is None:
            return list2
        elif list2 is None:
            return list1
        elif list1.val < list2.val:
            list1.next = self.mergeTwoLists(list1.next, list2)
            return list1
        else:
            list2.next = self.mergeTwoLists(list1, list2.next)
            return list2


    def removeDuplicatesFromSortedArray(self, nums: list[int]) -> int:
        """
        Solution to Remove Duplicates From Sorted Array (Easy)

        https://leetcode.com/problems/remove-duplicates-from-sorted-array

        """
        count = 1
        if len(nums) == 0:
            return 0
        for i in range(1, len(nums)):
            if nums[i] != nums[i-1]:
                nums[count] = nums[i]
                count += 1

        return count

    def removeElement(self, nums: list[int], val: int) -> int:
        count = 0
        for i in nums:
            if i != val:
                nums[count] = i
                count += 1
        return count

    def strStr(self, haystack: str, needle: str) -> int:
        """
        Solution for Implement strStr (Easy)

        https://leetcode.com/problems/implement-strstr

        """
        if needle not in haystack:
            return -1
        for i in range(len(haystack)):
            if needle in haystack[i:i+len(needle)]:
                return i

    def searchInsert(self, nums: list[int], target: int) -> int:
        """
        Solution for Search Insert Position

        https://leetcode.com/problems/search-insert-position

        algorithm should be written in O(log n) runtime complexity.
        """
        left, right = 0, len(nums) - 1
        while left <= right:
            pivot = (left + right) // 2
            if nums[pivot] == target:
                return pivot
            if target < nums[pivot]:
                right = pivot - 1 
            else:
                left = pivot + 1
        return left

    def maxSubArray(self, nums: list[int]) -> int:
        """
        Solution to Maximum Subarray (Easy) using Kadane's Algorithm.

        https://leetcode.com/problems/maximum-subarray
        """
        for i in range(1, len(nums)):
            nums[i] = max(nums[i], nums[i-1] + nums[i])

        return max(nums)

    def lengthOfLastWord(self, s: str) -> int:
        """
        Length of Last Word (Easy) Solution

        https://leetcode.com/problems/length-of-last-word
        """
        return len(s.split()[-1])
    def plusOne(self, digits: list[int]) -> list[int]:
        """
        Solution to Plus One (Easy)

        https://leetcode.com/problems/plus-one
        """
        s = ''.join([str(x) for x in digits])
        _s = int(s) + 1
        return [int(x) for x in str(_s)]

    def plusOneRecursion(self, digits: list[int]) -> list[int]:
        """
        Recursion Solution to Plus One (Easy)
        """
        length = len(digits) - 1
        while digits[length] == 9:
            digits[length] = 0
            length -= 1
        if (length < 0):
            digits = [1] + digits
        else:
            digits[length] += 1
        return digits

    def addBinary(self, a: str, b: str) -> str:
        """
        Solution to Add Binary (Easy)
        
        https://leetcode.com/problems/add-binary
        """
        x, y = int(a, 2), int(b, 2)
        while y:
            answer = x ^ y
            carry = (x & y) << 1
            x, y = answer, carry
        return bin(x)[2:]
    
    def mySqrt(self, x: int) -> int:
        """
        Solution to Sqrt (Easy)
        
        https://leetcode.com/problems/sqrtx
        """
        if x < 2:
            return x
        
        left = int(e**(0.5 * log(x)))
        right = left + 1
        return left if right * right > x else right

    def climbStairs(self, n: int) -> int:
        """
        Solution to Climbing Stairs (Easy) utilizing Fibonacci Series.
        
        https://leetcode.com/problems/climbing-stairs
        """
        x, y = 1, 1
        for i in range(n):
            x, y = y, x+y
        return x
    
    def removeDuplicatesFromLinkedList(self, head: ListNode) -> list:
        """
        Solution to Remove Duplicates from Sorted List (Easy)

        https://leetcode.com/problems/remove-duplicates-from-sorted-list/
        """
        current = head

        while current:
            while current.next and current.val == current.next.val:
                current.next = current.next.next
            current = current.next
        
        return head
    
    def mergeSortedArray(self, nums1: list[int], m: int, nums2: list[int], n: int) -> None:
        """
        Solution to Merge Sorted Array (Easy)

        https://leetcode.com/problems/merge-sorted-array
        """
        nums1[m:] = nums2[:n]
        nums1.sort()
    
    def mergeSortedArrayBruteForce(self, nums1: list[int], m: int, nums2: list[int], n: int) -> None:
        """
        Brute force solution to Merge Sorted Array (Easy)
        """
        for i in range(n):
            nums1[i+m] = nums2[i]
        
        nums1.sort()

    def binaryTreeInorderTraversalRecursive(self, root: TreeNode) -> list[int]:
        """
        Recursive Solution to Binary Tree Inorder Traversal (Easy)

        https://leetcode.com/problems/binary-tree-inorder-traversal
        """
        result = []

        def inorder(root):
            if not root:
                return
            inorder(root.left)
            result.append(root.val)
            inorder(root.right)
        
        inorder(root)
        return result

    def binaryTreeInorderTraversalIterative(self, root: TreeNode) -> list[int]:
        """
        Iterative Solution to Binary Tree Inorder Traversal (Easy)
        """

        result = []
        stack = []
        current = root

        while current or stack:
            while current:
                stack.append(current)
                current = current.left
            current = stack.pop()
            result.append(current.val)
            current = current.right
        
        return result
    
    def sameTree(self, p: TreeNode, q: TreeNode) -> bool:
        """
        Solution to Same Tree (Easy) using Depth First Search method.

        https://leetcode.com/problems/same-tree
        """
        # Check if root node is the same.
        if not p and not q: # If both are null, reture True
            return True
        if (not p or not q) or p.val != q.val:
            return False # If one is null and other is not, return False, also if the value is not the same, return False.

        return (self.isSameTree(p.left, q.left) and
            self.isSameTree(p.right, q.right))

    def isSymmetric(self, root: TreeNode) -> bool:
        """
        Solution to Symmetric Tree (Easy)
        
        https://leetcode.com/problems/symmetric-tree
        """
        if root is None:
            return True
        
        def ismirror(left, right):
            if left and right:
                return left.val == right.val and ismirror(left.left, right.right) and ismirror(left.right, right.left)
            return left == right

        return ismirror(root.left, root.right)

    def maxDepthRecursion(self, root: TreeNode) -> int:
        """
        Recursion method for solving Maximum Depth of Binary Tree (Easy)

        https://leetcode.com/problems/maximum-depth-of-binary-tree
        """
        if not root:
            return 0
        
        return 1 + max(self.maxDepthRecursion(root.left), self.maxDepthRecursion(root.right))
    
    def maxDepthBreathFirstSearch(self, root: TreeNode) -> int:
        """
        Breath First Search for solving Maximum Depth of Binary Tree (Easy)
        """
        if not root:
            return 0
        
        queue = deque([root])
        depth = 0
        while queue:
            for i in range(len(queue)):
                node = queue.popleft()
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            
            depth += 1
        return depth
    
    def maxDepthIteration(self, root: TreeNode) -> int:
        """
        Iteration method for solving Maximum Depth of Binary Tree (Easy)
        """
        if not root:
            return 0
        
        stack = [[root, 1]]
        result = 1
        while stack:
            node, depth = stack.pop()

            if node:
                result = max(result, depth)
                stack.append([node.left, depth + 1])
                stack.append([node.right, depth + 1])

        return result
    
    def minDepth(self, root: TreeNode) -> int:
        """
        Minimum Depth of Binary Tree using Depth First Search (DFS) (Easy)

        https://leetcode.com/problems/minimum-depth-of-binary-tree/
        """
        if not root: return 0

        self.small = float('inf')

        def dfs(node, num):
            if not node: return
            if not node.left and not node.right:
                self.small = min(self.small, num)
            
            dfs(node.left, num + 1)
            dfs(node.right, num + 1)
        
        dfs(root, 1)
        return self.small

    def sortedArrayToBST(self, nums: list[int]) -> TreeNode:
        """
        Solution for Convert Sorted Array to Binary Search Tree (Easy)

        Time Complexity; O(n)

        https://leetcode.com/problems/converted-sorted-array-to-binary-search-tree/
        """

        def helper(left, right):
            if left > right:
                return None
            mid = (left + right) // 2
            root = TreeNode(nums[mid])
            root.left = helper(left, mid - 1)
            root.right = helper(mid + 1, right)
            return root
        
        return helper(0, len(nums) - 1)
    
    def isBalanced(self, root: TreeNode) -> bool:
        """
        Solution for Balanced Binary Tree (Easy)

        https://leetcode.com/problems/balanced-binary-tree
        """
        def helper(root):
            if not root:
                return 0
            left = helper(root.left)
            right = helper(root.right)
            if left == -1 or right == -1 or abs(left - right) > 1:
                return -1
            return max(left, right) + 1
        
        return helper(root) != -1
    
    def hasPathSum(self, root: TreeNode, targetSum: int) -> bool:
        """
        Solution for Path Sum (Easy)

        https://leetcode.com/problems/path-sum
        """
        sums = []

        def dsf(root, runningSum: int, sums: list):
            if root is None:
                return
            
            newRunningSum = runningSum + root.val
            if root.left is None and root.right is None:
                sums.append(newRunningSum)
                return
            
            dsf(root.left, newRunningSum, sums)
            dsf(root.right, newRunningSum, sums)
        
        dsf(root, 0, sums)
        return targetSum in sums

    def pascalTriangle(self, numRows: int) -> list[list[int]]:
        triangle = []
        for row_num in range(numRows):
            row = [None for _ in range(row_num + 1)]
            row[0], row[-1] = 1, 1
            for j in range(1, len(row) - 1):
                row[j] = triangle[row_num - 1][j - 1] + triangle[row_num - 1][j]
            triangle.append(row)
        return triangle

    def getRow(self, rowIndex: int) -> list[int]:
        row = [1]
        for i in range(rowIndex):
            for j in range(i, 0, -1):
                row[j] = row[j] + row[j-1]
            row.append(1)
        return row
    
    def singleNumber(self, nums: list[int]) -> int:
        no_duplicate_list = []
        for i in nums:
            if i not in no_duplicate_list:
                no_duplicate_list.append(i)
            else:
                no_duplicate_list.remove(i)
        
        return no_duplicate_list.pop()
    
    def hasCycle(self, head: ListNode) -> bool:
        nodes_seen = set()
        while head is not None:
            if head in nodes_seen:
                return True
            nodes_seen.add(head)
            head = head.next
        
        return False
    
    def preorderTraversalIteration(self, root: ListNode) -> list[int]:
        if root is None:
            return []
        
        stack, output = [root, ], []

        while stack:
            root = stack.pop()
            if root is not None:
                output.append(root.val)
                if root.right is not None:
                    stack.append(root.right)
                if root.left is not None:
                    stack.append(root.left)
                
        return output
    
    def preorderTraversalMorris(self, root: ListNode) -> list[int]:
        output = []
        while root is not None:
            if root.left is None:
                output.append(root.val)
                root = root.right
            else:
                pre = root.left
                while pre.right is not None and pre.right != root:
                    pre = pre.right
                
                if pre.right is None:
                    pre.right = root
                    output.append(root.val)
                    root = root.left
                else:
                    pre.right = None
                    root = root.right
        
        return output