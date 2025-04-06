import random
import tkinter
import torch
import time

#设置随机种子
random.seed(123)

# 参数设置
ALPHA = 1.0  # 信息素启发因子
BETA = 2.0   # 期望启发因子
RHO = 0.5    # 信息素挥发系数
Q = 100.0    # 信息素释放总量
city_num = 50  # 城市数量
ant_num = 50   # 蚂蚁数量
iterations = 100  # 最大迭代次数

file_path = '../../data/data/visual_friendly_tsp50.pkl'
data = torch.load(file_path)
# data = torch.load(file_path, map_location='cpu')  # 确保加载到CPU
# 提取坐标数据
all_coords = data['x'].numpy()  # 形状 (1000, 20, 2)

# 选择第一个TSP20实例
distance_x = all_coords[0][:, 0].tolist()
distance_y = all_coords[0][:, 1].tolist()

# 计算城市之间的距离
distance_graph = [[0.0 for _ in range(city_num)] for _ in range(city_num)]
pheromone_graph = [[1.0 for _ in range(city_num)] for _ in range(city_num)]
for i in range(city_num):
    for j in range(city_num):
        distance = ((distance_x[i] - distance_x[j]) ** 2 + (distance_y[i] - distance_y[j]) ** 2) ** 0.5
        distance_graph[i][j] = float(int(distance + 0.5))

# 路径距离计算函数
def calculate_distance(path):
    return sum(distance_graph[path[i]][path[i + 1]] for i in range(-1, city_num - 1))

# 2-Opt 局部搜索优化
def two_opt(path):
    best_path = path[:]
    best_distance = sum(distance_graph[best_path[i]][best_path[i+1]] for i in range(-1, city_num-1))
    for i in range(1, len(path) - 2):
        for j in range(i + 1, len(path)):
            new_path = path[:]
            new_path[i:j] = reversed(new_path[i:j])
            new_distance = sum(distance_graph[new_path[k]][new_path[k+1]] for k in range(-1, city_num-1))
            if new_distance < best_distance:
                best_path, best_distance = new_path, new_distance
    return best_path

# 蚂蚁类
class Ant:
    def __init__(self):
        self.path = []
        self.total_distance = 0.0
        self.visited = set()

    def search_path(self):
        self.path = [random.randint(0, city_num - 1)]
        self.visited = set(self.path)
        self.total_distance = 0.0

        while len(self.path) < city_num:
            next_city = self.select_next_city()
            self.path.append(next_city)
            self.visited.add(next_city)

        self.total_distance = sum(distance_graph[self.path[i]][self.path[i+1]] for i in range(-1, city_num-1))

    def select_next_city(self):
        current_city = self.path[-1]
        probabilities = []
        for city in range(city_num):
            if city not in self.visited:
                pheromone = pheromone_graph[current_city][city] ** ALPHA
                heuristic = (1.0 / distance_graph[current_city][city]) ** BETA
                probabilities.append(pheromone * heuristic)
            else:
                probabilities.append(0)

        total_prob = sum(probabilities)
        if total_prob == 0: return random.choice(list(set(range(city_num)) - self.visited))
        r = random.uniform(0, total_prob)
        for i, prob in enumerate(probabilities):
            r -= prob
            if r <= 0:
                return i

# TSP 主类
class TSP:
    def __init__(self, root):
        self.run_time = 0.0  # 新增运行时间记录属性
        self.root = root
        self.canvas = tkinter.Canvas(root, width=800, height=800, bg="white")
        self.canvas.pack()
        self.best_path = None
        self.best_distance = float('inf')

        # 绘制城市节点
        self.nodes = [(distance_x[i], distance_y[i]) for i in range(city_num)]
        for x, y in self.nodes:
            self.canvas.create_oval(x - 5, y - 5, x + 5, y + 5, fill="red")

    def run(self):
        start_time = time.perf_counter()  # 记录开始时间
        global ALPHA, BETA
        ants = [Ant() for _ in range(ant_num)]

        for iteration in range(iterations):
            for ant in ants:
                ant.search_path()
                if ant.total_distance < self.best_distance:
                    self.best_distance = ant.total_distance
                    self.best_path = two_opt(ant.path)

            self.update_pheromone(ants)
            self.draw_path()
            self.root.update()  # 更新UI

            ALPHA *= 1.01  # 动态调整参数
            BETA *= 0.99
            print(f"Iteration {iteration+1}, Best Distance: {self.best_distance}")

        end_time = time.perf_counter()  # 记录结束时间
        self.run_time = end_time - start_time  # 计算总耗时
        print("Best Path:", self.best_path)
        print("Best Distance:", self.best_distance)
        print(f"Algorithm time: {self.run_time:.2f} seconds")
        print(f"Iteration rate: {iterations / self.run_time:.1f} Iterations/second")

    def update_pheromone(self, ants):
        for i in range(city_num):
            for j in range(city_num):
                pheromone_graph[i][j] *= (1 - RHO)

        for ant in ants:
            for i in range(-1, city_num-1):
                start, end = ant.path[i], ant.path[i+1]
                pheromone_graph[start][end] += Q / ant.total_distance

    def draw_path(self):
        self.canvas.delete("path")
        if self.best_path:
            for i in range(-1, city_num-1):
                start, end = self.best_path[i], self.best_path[i+1]
                x1, y1 = self.nodes[start]
                x2, y2 = self.nodes[end]
                self.canvas.create_line(x1, y1, x2, y2, fill="blue", tags="path")

# 主函数
if __name__ == "__main__":
    root = tkinter.Tk()
    root.title("Ant Colony Optimization for TSP")
    tsp = TSP(root)
    tsp.run()
    root.mainloop()