# FedAvg demo1 run
# 2025 04 26

from lib_demo1 import SimpleMLP, DataLoader, test_dataset, num_clients, copy, train_client, client_datasets, aggregate_weights, test_global_model

# 初始化全局模型
global_model = SimpleMLP()

# 联邦学习参数
num_rounds = 10  # 通信轮次
client_epochs = 1
batch_size = 32

# 测试数据加载器
test_loader = DataLoader(test_dataset, batch_size=1024)

for round in range(num_rounds):
    print(f"\n=== Round {round + 1} ===")

    # 各客户端下载全局模型并本地训练
    client_weights = []
    for client_id in range(num_clients):
        local_model = copy.deepcopy(global_model)
        local_weights = train_client(local_model, client_datasets[client_id], epochs=client_epochs,
                                     batch_size=batch_size)
        client_weights.append(local_weights)

    # 服务器聚合参数
    global_weights = aggregate_weights(client_weights)
    global_model.load_state_dict(global_weights)

    # 测试全局模型
    test_global_model(global_model, test_loader)
