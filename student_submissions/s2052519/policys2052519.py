import numpy as np

from policy import Policy
from scipy.optimize import linprog

class FirstCutPolicy:
    def get_action(self, observation, info):
        stocks = observation['stocks']
        products = observation['products']
        
        for product in products: # Needs optimization for runtime
            product_size = product['size']
            demand = product['quantity']
            
            for stock_idx, stock in enumerate(stocks):
                stock_height, stock_width = stock.shape
                product_height, product_width = product_size
                
                if product_height <= stock_height and product_width <= stock_width and demand > 0:
                    for i in range(stock_height - product_height + 1): # O(n^2) -> Needs optimization
                        for j in range(stock_width - product_width + 1):
                            subgrid = stock[i:i + product_height, j:j + product_width]
                            if np.all(subgrid == -1):
                                action = {
                                    "stock_idx": stock_idx,
                                    "size": (product_height, product_width),
                                    "position": (i, j),
                                }

                                product['quantity'] -= 1
                                return action

        return None

### WIP
class ColumnGeneration(Policy):
    def __init__(self):
        super().__init__()
        self.patterns = []
        self.initialized = False
        self.num_products = 0

    def get_action(self, observation, info):
        products = observation["products"]
        stocks = observation["stocks"]

        demand = np.array([product["quantity"] for product in products])
        sizes = [product["size"] for product in products]
        num_products = len(products)

        if not self.initialized or self.num_products != num_products:
            self.initialize_patterns(num_products, sizes, stocks)
            self.initialized = True
            self.num_products = num_products

        while True:
            c = np.ones(len(self.patterns))
            A = np.array(self.patterns).T
            b = demand

            res = linprog(c, A_ub=-A, b_ub=-b, bounds=(0, None), method='highs')

            if res.status != 0:
                break

            dual_prices = getattr(res, "ineqlin", {}).get("marginals", None)

            if dual_prices is None:
                break

            new_pattern = self.generate_pattern(dual_prices, sizes, stocks)

            if new_pattern is None or any(np.array_equal(new_pattern, p) for p in self.patterns):
                break

            self.patterns.append(new_pattern)

        best_pattern = self.choose_best_pattern(self.patterns, demand)
        action = self.convert_pattern_to_action(best_pattern, sizes, stocks)
        return action

    def initialize_patterns(self, num_products, sizes, stocks):
        """Generate initial patterns based on product sizes and stock availability."""
        self.patterns = []
        for stock in stocks:
            stock_size = self._get_stock_size_(stock)
            for i in range(num_products):
                if stock_size[0] >= sizes[i][0] and stock_size[1] >= sizes[i][1]:
                    pattern = np.zeros(num_products, dtype=int)
                    pattern[i] = 1
                    self.patterns.append(pattern)

        self.patterns = list({tuple(p): p for p in self.patterns}.values())

    def generate_pattern(self, dual_prices, sizes, stocks):
        best_pattern = None
        best_cost = float('-inf')

        for stock in stocks:
            stock_width, stock_height = self._get_stock_size_(stock)

            dp = np.zeros((stock_height + 1, stock_width + 1))
            pattern = np.zeros(len(sizes), dtype=int)

            for i, size in enumerate(sizes):
                prod_width, prod_height = size
                if prod_width <= stock_width and prod_height <= stock_height and dual_prices[i] > 0:
                    for x in range(stock_width, prod_width - 1, -1):
                        for y in range(stock_height, prod_height - 1, -1):
                            dp[y][x] = max(dp[y][x], dp[y - prod_height][x - prod_width] + dual_prices[i])

            width, height = stock_width, stock_height
            for i in range(len(sizes) - 1, -1, -1):
                prod_width, prod_height = sizes[i]
                if width >= prod_width and height >= prod_height and dp[height][width] == dp[height - prod_height][width - prod_width] + dual_prices[i]:
                    pattern[i] += 1
                    width -= prod_width
                    height -= prod_height

            reduced_cost = np.dot(pattern, dual_prices) - 1
            if reduced_cost > best_cost:
                best_cost = reduced_cost
                best_pattern = pattern

        return best_pattern if best_cost > 1e-6 else None

    def choose_best_pattern(self, patterns, demand):
        """Select the pattern that satisfies the most demand."""
        best_pattern = None
        max_coverage = -1

        for pattern in patterns:
            coverage = np.sum(np.minimum(pattern, demand))
            if coverage > max_coverage:
                max_coverage = coverage
                best_pattern = pattern

        return best_pattern

    def convert_pattern_to_action(self, pattern, sizes, stocks):
        """Translate a pattern into an actionable cutting plan."""
        for i, count in enumerate(pattern):
            if count > 0:
                product_size = sizes[i]
                for stock_idx, stock in enumerate(stocks):
                    stock_width, stock_height = self._get_stock_size_(stock)
                    if stock_width >= product_size[0] and stock_height >= product_size[1]:
                        position = self.bottom_left_placement(stock, product_size)
                        if position:
                            return {"stock_idx": stock_idx, "size": product_size, "position": position}

        return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}

    def bottom_left_placement(self, stock, product_size):
        """Find the bottom-left most placement for a product in the stock."""
        stock_width, stock_height = self._get_stock_size_(stock)
        prod_width, prod_height = product_size

        for y in range(stock_height - prod_height + 1):
            for x in range(stock_width - prod_width + 1):
                if self._can_place_(stock, (x, y), product_size):
                    return x, y

        return None
    

class Policy2052519(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"

        # Student code here
        self.policy = None
        if policy_id == 1:
            self.policy = FirstCutPolicy()
        elif policy_id == 2:
            self.policy = ColumnGeneration()

    def get_action(self, observation, info):
        # Student code here
        if self.policy is not None:
            return self.policy.get_action(observation, info)
        else:
            raise NotImplementedError("Policy is not implemented yet.")

    # Student code here
    # You can add more functions if needed