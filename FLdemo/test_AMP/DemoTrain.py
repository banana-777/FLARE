# 05 02
# 执行训练

import time
from datetime import datetime
from DemoLibrary import *

if __name__ == "__main__":
    # 开始计时
    start_timestamp = time.time()
    start_time = datetime.fromtimestamp(start_timestamp).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    print(f"任务开始时间: {start_time}")

    amp_status = "启用" if FLConfig.AMP_ENABLED else "禁用"
    print(f"\n自动混合精度训练（AMP）状态: {amp_status}")

    # 开始训练
    trainer = FLTrainer()
    trainer.train()

    # 结束计时
    end_timestamp = time.time()
    end_time = datetime.fromtimestamp(end_timestamp).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    duration_seconds = round(end_timestamp - start_timestamp, 3)
    hours = int(duration_seconds // 3600)
    minutes = int((duration_seconds % 3600) // 60)
    seconds = round(duration_seconds % 60, 3)
    print(f"任务结束时间: {end_time}")
    print(f"\n总耗时: {duration_seconds} 秒")
    print(f"等价于: {hours:02d}:{minutes:02d}:{seconds:06.3f} (时:分:秒.毫秒)")

    # 绘图
    visualizer = TrainingVisualizer(FLConfig.LOG_FILE)
    visualizer.plot()
    visualizer.save_plot("training_plot_1.png")
    # GUI使用示例
    # gui_model = FLGUIInterface.load_model("./fl_models/final_model.pt")
    # prediction = FLGUIInterface.predict(gui_model, test_image)
