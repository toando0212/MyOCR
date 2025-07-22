import sys

def main():
    n, k = map(int, sys.stdin.readline().split())
    projects = [int(sys.stdin.readline()) for _ in range(n)]

    prefix_sum = [0] * (n + 1)
    for i in range(n):
        prefix_sum[i + 1] = prefix_sum[i] + projects[i]

    dp = [0] * (n + 1)

    for i in range(1, n + 1):
        dp[i] = dp[i-1]

        for j in range(1, k + 1):
            if i - j >= 0:
                current_block_profit = prefix_sum[i] - prefix_sum[i-j]

                profit_before_block = dp[i-j-1] if i - j - 1 >= 0 else 0
                dp[i] = max(dp[i], profit_before_block + current_block_profit)

    print(dp[n])

main()