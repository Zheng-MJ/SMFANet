import os
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 设置 Times New Roman 字体和默认字体大小
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 15

# 创建图形和轴
fig, ax = plt.subplots(figsize=(15, 10))

# 定义数据字典列表，包含方法名、x 坐标、y 坐标、FLOPs
data = [
    {'method': 'SMFANet',        'x': 197,  'y': 30.82, 'flops': 11},
    {'method': 'SMFANet+',       'x': 496,  'y': 31.29, 'flops': 28},
    {'method': 'EDSR-baseline',  'x': 1518, 'y': 30.35, 'flops': 114},
    {'method': 'LAPAR-A',        'x': 659,  'y': 30.42, 'flops': 94},
    {'method': 'IMDN',           'x': 715,  'y': 30.45, 'flops': 41},
    {'method': 'CARN',           'x': 1592, 'y': 30.47, 'flops': 91},
    {'method': 'ShuffleMixer',   'x': 411,  'y': 30.65, 'flops': 28},
    {'method': 'NGswin',         'x': 1019, 'y': 30.80, 'flops': 40},
    {'method': 'HPINet-S',       'x': 463,  'y': 30.92, 'flops': 88},
    {'method': 'SPIN',           'x': 555,  'y': 30.98, 'flops': 42},
    {'method': 'ELAN-light',     'x': 640,  'y': 30.92, 'flops': 54},
    {'method': 'SwinIR-light',   'x': 930,  'y': 30.92, 'flops': 65},
    {'method': 'SRFormer-light', 'x': 873,  'y': 31.17, 'flops': 63}
]

colors =  ['#990000',  '#ff0000', '#264653', '#9933FF', '#33CC33',  '#336699', '#FF6666', '#666666', '#FFFF66', '#003366', '#333399', '#000000', '#999933']

# 定义 main 函数
def main(save_dir = "plt"):
    # 绘制散点图
    for d, color in zip(data, colors):
        area = 3 * (d['flops'] ** 2)
        ax.scatter(d['x'], d['y'], s=area, alpha=0.8, marker='.', c=color, edgecolors='white', linewidths=2.0)
        # 使用 Times New Roman 字体
        font_props = fm.FontProperties('Times New Roman')
        ax.annotate(d['method'], (d['x'], d['y']), xytext=(0, 10), textcoords='offset points',
                    fontproperties=font_props, fontsize=20, ha='right')

    # 网格线
    ax.grid(True)

    # 添加红色的虚线
    ax.plot([data[0]['x'], data[1]['x']], [data[0]['y'], data[1]['y']], 
            color='red', linestyle='--', linewidth=1)

    # 设置x轴和y轴的范围
    ax.set_xlim(100, 1700)
    ax.set_ylim(30.25, 31.35)

    # 设置x轴和y轴的刻度
    ax.set_xticks([400, 800, 1200, 1600])
    ax.set_xticklabels([400, 800, 1200, 1600], fontproperties=font_props, size=30)
    ax.set_yticks([30.40, 30.60, 30.80, 31.0, 31.20])
    ax.set_yticklabels([30.40, 30.60, 30.80, 31.0, 31.20], fontproperties=font_props, size=30)

    # 设置轴标签和标题
    ax.set_ylabel('PSNR (dB)', fontproperties=font_props, fontsize=35)
    ax.set_xlabel('Parameters (K)', fontproperties=font_props, fontsize=35)
    plt.suptitle('PSNR vs. Parameters vs. FLOPs', fontproperties=font_props, fontsize=35)


    plt.savefig(os.path.join(save_dir, 'model_complexity.pdf'), bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    
    main()