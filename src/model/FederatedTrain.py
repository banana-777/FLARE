# 04 29
# 执行训练

from FederatedLibrary import *

if __name__ == "__main__":
    trainer = FLTrainer()
    trainer.train()

    # GUI使用示例
    # gui_model = FLGUIInterface.load_model("./fl_models/final_model.pt")
    # prediction = FLGUIInterface.predict(gui_model, test_image)
