import onnxruntime as ort
import numpy as np
from PIL import Image
import json
import os

class_names =['万', '东', '丝', '两', '串', '丸', '丽', '之', '乌', '乐', '乙', '九', '乳', '乾', '五', '井', '人', '仔', '佛', '佳', '保', '兔', '兰', '兴', '其', '养', '农', '冬', '冻', '凉', '凝', '凤', '凯', '刀', '切', '列', '利', '制', '剁', '剂', '加', '勾', '包', '化', '升', '卜', '卤', '卷', '原', '去', '叉', '双', '发', '古', '叶', '司', '合', '吉', '君', '和', '咖', '咸', '哈', '响', '啡', '团', '园', '固', '圆', '圣', '圪', '块', '培', '基', '堡', '塔', '墨', '壳', '复', '多', '大', '太', '夫', '头', '夹', '姜', '嫩', '安', '定', '宫', '容', '富', '尝', '尾', '层', '岗', '川', '州', '巴', '市', '希', '廉', '御', '徽', '忌', '成', '扁', '扇', '手', '扒', '抄', '抓', '抗', '拉', '拔', '拨', '排', '摩', '撕', '散', '斋', '斑', '方', '无', '日', '早', '旭', '旺', '春', '晶', '暑', '曼', '月', '木', '末', '朱', '李', '杞', '条', '杨', '杯', '杷', '松', '极', '枇', '果', '枝', '枣', '柚', '柳', '柿', '样', '核', '格', '桂', '桃', '桧', '梅', '椒', '楼', '榄', '榴', '槟', '樱', '橄', '橘', '橙', '欧', '正', '段', '毋', '毛', '毫', '气', '氯', '水', '氽', '汁', '江', '汤', '沙', '油', '泡', '泥', '洋', '派', '浦', '浪', '海', '消', '溶', '滑', '火', '炊', '炒', '炖', '炝', '炸', '烤', '烧', '烩', '焖', '煎', '煲', '燕', '爆', '片', '牌', '牛', '玉', '玛', '玫', '环', '珍', '珠', '理', '琉', '瑰', '璃', '瓜', '甜', '生', '用', '田', '甲', '番', '白', '百', '皮', '盒', '盔', '盖', '盘', '真', '眼', '着', '石', '砂', '破', '硅', '硫', '磷', '磺', '神', '祥', '福', '科', '窝', '竟', '竹', '筋', '筒', '箕', '簸', '米', '籽', '粉', '粑', '粘', '粥', '粽', '精', '糌', '糕', '糖', '糯', '素', '紫', '红', '绿', '罐', '羊', '美', '羹', '翅', '老', '耳', '联', '肉', '肝', '肺', '胜', '胡', '胶', '脂', '脆', '腌', '腐', '腿', '艇', '色', '芋', '芝', '芥', '花', '芳', '芹', '苯', '苹', '茄', '茭', '茴', '茶', '草', '荔', '荷', '莫', '菌', '菜', '菠', '萄', '萝', '萨', '葡', '葱', '蒸', '蓉', '蔓', '蔬', '蕉', '薇', '藏', '藕', '藤', '虎', '虾', '蛋', '蜀', '蜜', '螺', '血', '解', '豆', '豌', '贡', '贵', '越', '蹄', '软', '辣', '达', '连', '迪', '速', '酒', '酥', '酪', '酱', '酶', '酸', '酿', '醇', '醉', '醚', '里', '金', '钙', '钱', '钾', '铃', '银', '铺', '锅', '锦', '锰', '镜', '阳', '阿', '雕', '露', '青', '面', '风', '食', '餐', '饭', '饮', '饵', '饺', '饼', '馅', '馆', '馏', '馕', '香', '马', '骨', '魂', '鱼', '鲜', '鲤', '鲫', '鳝', '鸠', '鸡', '鸭', '鸿', '鹅', '鹿', '麦', '麻', '黄', '黑', '鼎', '鼓', '龙']

class ONNXPredictor:
    def __init__(self, onnx_path):
        # 初始化ONNX推理会话[4,7](@ref)
        self.session = ort.InferenceSession(onnx_path)
        self.input_name = self.session.get_inputs()[0].name

        # 验证输入数据类型（网页4、网页7）
        input_type = self.session.get_inputs()[0].type
        assert "float" in input_type, f"模型需要float32输入，当前类型：{input_type}"

        self.class_names = class_names
        # 预处理参数（需与训练一致）
        self.input_size = (128, 128)
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def preprocess(self, image_path):
        """图像预处理流程（强制float32转换）[6,9](@ref)"""
        img = Image.open(image_path).convert('RGB')
        img = img.resize(self.input_size)

        # 转换为float32并归一化[5](@ref)
        img = np.array(img).astype(np.float32) / 255.0
        img = (img - self.mean) / self.std

        # 转换为NCHW格式
        img = img.transpose(2, 0, 1)  # HWC -> CHW
        img = np.expand_dims(img, axis=0)  # 添加batch维度
        return img

    def predict(self, image_path):
        """执行推理并返回最大概率标签"""
        input_tensor = self.preprocess(image_path)

        # ONNX推理[5,10](@ref)
        outputs = self.session.run(None, {self.input_name: input_tensor})
        logits = outputs[0]

        # 获取最大概率索引[9](@ref)
        pred_idx = np.argmax(logits, axis=1)[0]
        return self.class_names[pred_idx]


# -------------------- 使用示例 --------------------
if __name__ == "__main__":
    # 初始化预测器（假设模型和标签文件路径正确）
    predictor = ONNXPredictor(
        onnx_path="best_model_epoch5.onnx",
    )

    # 执行预测
    # test_image = "test_images/1a1cbb7fa8de407b97ff16f606669f1e.png"
    # test_image = "test_images/68e5624589364f5cb520fbc200af3b4c.png"
    # test_image = "test_images/24558071e03449a1971628816cfae3dd.png"
    test_image = "test_images/ef91b6296f67438490595a221ff61ae9_.png"
    print(f"预测结果：{predictor.predict(test_image)}")