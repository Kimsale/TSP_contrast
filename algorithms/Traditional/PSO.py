import random
import tkinter
import torch
import time

#设置随机种子
random.seed(123)

# 参数设置
city_num = 50
particle_num = 100
iterations = 100
w_max = 0.9  # 惯性权重最大值
w_min = 0.4  # 惯性权重最小值
c1 = 2.0  # 个人学习因子
c2 = 2.0  # 社会学习因子


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
for i in range(city_num):
    for j in range(city_num):
        distance_graph[i][j] = ((distance_x[i] - distance_x[j]) ** 2 + (distance_y[i] - distance_y[j]) ** 2) ** 0.5

# 路径距离计算函数
def calculate_distance(path):
    return sum(distance_graph[path[i]][path[i + 1]] for i in range(-1, city_num - 1))

# 2-Opt 局部搜索
def two_opt(path):
    best_path = path[:]
    best_distance = calculate_distance(path)
    for i in range(1, city_num - 1):
        for j in range(i + 1, city_num):
            new_path = path[:]
            new_path[i:j] = reversed(new_path[i:j])
            new_distance = calculate_distance(new_path)
            if new_distance < best_distance:
                best_path, best_distance = new_path, new_distance
    return best_path

# 粒子类
class Particle:
    def __init__(self):
        self.path = random.sample(range(city_num), city_num)
        self.p_best = self.path[:]
        self.p_best_distance = calculate_distance(self.path)
        self.velocity = []

    def update(self, g_best, w):
        new_path = self.path[:]
        for i in range(city_num):
            if random.random() < w:
                new_path[i] = g_best[i]
            elif random.random() < c1:
                new_path[i] = self.p_best[i]
        # 修复路径合法性
        self.path = self.repair_path(new_path)
        self.path = two_opt(self.path)  # 2-Opt 局部优化
        distance = calculate_distance(self.path)
        if distance < self.p_best_distance:
            self.p_best = self.path[:]
            self.p_best_distance = distance

    def repair_path(self, path):
        visited = set()
        new_path = []
        for city in path:
            if city not in visited:
                visited.add(city)
                new_path.append(city)
        missing_cities = set(range(city_num)) - visited
        new_path.extend(missing_cities)
        return new_path

# PSO类
class PSO:
    def __init__(self):
        self.particles = [Particle() for _ in range(particle_num)]
        self.g_best = min(self.particles, key=lambda p: p.p_best_distance).p_best
        self.g_best_distance = calculate_distance(self.g_best)

    def run(self, canvas, draw_path, root):
        start_time = time.perf_counter()  # 记录开始时间
        for t in range(iterations):
            w = w_max - t * (w_max - w_min) / iterations  # 动态调整惯性权重
            for particle in self.particles:
                particle.update(self.g_best, w)
                if particle.p_best_distance < self.g_best_distance:
                    self.g_best = particle.p_best[:]
                    self.g_best_distance = particle.p_best_distance

            draw_path(self.g_best)
            root.update()
            print(f"Iteration {t+1}, Best Distance: {self.g_best_distance}")
        end_time = time.perf_counter()  # 记录结束时间
        total_time = end_time - start_time
        return self.g_best, self.g_best_distance, total_time

# 可视化类
class TSPVisualizer:
    def __init__(self, root):
        self.root = root
        self.canvas = tkinter.Canvas(root, width=800, height=800, bg="white")
        self.canvas.pack()
        self.nodes = [(distance_x[i], distance_y[i]) for i in range(city_num)]
        self.pso = PSO()
        self.draw_nodes()

    def draw_nodes(self):
        """ 绘制城市节点 """
        for x, y in self.nodes:
            self.canvas.create_oval(x - 5, y - 5, x + 5, y + 5, fill="red")

    def draw_path(self, path):
        """ 绘制路径 """
        self.canvas.delete("path")
        for i in range(-1, city_num - 1):
            x1, y1 = self.nodes[path[i]]
            x2, y2 = self.nodes[path[i + 1]]
            self.canvas.create_line(x1, y1, x2, y2, fill="blue", tags="path")

    def run(self):
        """ 运行PSO算法并实时可视化 """
        best_path, best_distance, run_time = self.pso.run(self.canvas, self.draw_path, self.root)
        self.draw_path(best_path)
        print("Final Best Path:", best_path)
        print("Final Best Distance:", best_distance)
        print(f"Algorithm time: {run_time:.2f} seconds")
        print(f"Iteration rate: {iterations / run_time:.1f} Iterations/second")

# 主函数
if __name__ == "__main__":
    root = tkinter.Tk()
    root.title("Particle Swarm Optimization for TSP")
    tsp = TSPVisualizer(root)
    tsp.run()
    root.mainloop()
