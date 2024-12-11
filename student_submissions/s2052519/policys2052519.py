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
class ColumnGenerationPolicy:
    def __init__(self):
        self.patterns = []
        self.demands = None
        self.stock = None

    def initialize(self, observation):
        self.demands = observation['products']
        self.stock = observation['stocks']

        self.patterns = []
        for product in self.demands:
            self.patterns.append({
                "stock_idx": 0,
                "size": product["size"],
                "positions": [(0, 0)]
            })

    def solve_master_problem(self):
        c = np.ones(len(self.patterns))

        A_eq = np.array(self.patterns).T
        b_eq = np.array([product['quantity'] for product in self.demands])

        res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=(0, None), method="highs")

        return res.x, res.fun

    def solve_pricing_problem(self, duals):
        c = np.ones(len(self.patterns))

        A_eq = []
        b_eq = [product["quantity"] for product in self.demands]
        for product_idx, product in enumerate(self.demands):
            row = []
            for pattern in self.patterns:
                if np.array_equal(pattern["size"], product["size"]):
                    row.append(1)
                else:
                    row.append(0)
            A_eq.append(row)

        res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=(0, None), method="highs")
        return res.x, res.fun

    def solve_pricing_problem(self, duals):
        new_pattern = None
        max_value = float("-inf")

        for stock_idx, stock in enumerate(self.stocks):
            stock_height, stock_width = stock.shape

            for product_idx, product in enumerate(self.demands):
                product_height, product_width = product["size"]

                for i in range(stock_height - product_height + 1):
                    for j in range(stock_width - product_width + 1):
                        subgrid = stock[i:i + product_height, j:j + product_width]
                        if np.all(subgrid == -1):
                            value = duals[product_idx]
                            if value > max_value:
                                max_value = value
                                new_pattern = {
                                    "stock_idx": stock_idx,
                                    "size": product["size"],
                                    "positions": [(i, j)]
                                }

        return new_pattern

    def get_action(self, observation, info):
        if not self.patterns:
            self.initialize(observation)

        solution, _ = self.solve_master_problem()

        duals = solution
        new_pattern = self.solve_pricing_problem(duals)

        if new_pattern:
            self.patterns.append(new_pattern)
            return {
                "stock_idx": new_pattern["stock_idx"],
                "size": tuple(new_pattern["size"]),
                "position": new_pattern["positions"][0]
            }
        else:
            return None
    

class Policy2052519(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"

        # Student code here
        self.policy = None
        if policy_id == 1:
            self.policy = FirstCutPolicy()
        elif policy_id == 2:
            pass

    def get_action(self, observation, info):
        # Student code here
        if self.policy is not None:
            return self.policy.get_action(observation, info)
        else:
            raise NotImplementedError("Policy ID 2 is not implemented yet.")

    # Student code here
    # You can add more functions if needed