import os
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 设置 Times New Roman 字体和默认字体大小
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 15

# 创建图形和轴
fig, ax = plt.subplots(figsize=(15, 10))

# 定义数据字典列表
data = [
    {'method': 'SMFANet',        'x': 9.26,   'y': 30.82, 'Mem': 93.20},
    {'method': 'SMFANet+',       'x': 17.39,  'y': 31.29, 'Mem': 247.51},
    {'method': 'EDSR-baseline',  'x': 15.21,  'y': 30.35, 'Mem': 507.13},
    {'method': 'LAPAR-A',        'x': 24.40,  'y': 30.42, 'Mem': 1811.46},
    {'method': 'IMDN',           'x': 9.64,   'y': 30.45, 'Mem': 204.27},
    {'method': 'CARN',           'x': 13.54,  'y': 30.47, 'Mem': 702.07},
    {'method': 'ShuffleMixer',   'x': 19.21,  'y': 30.65, 'Mem': 474.79},
    {'method': 'NGswin',         'x': 108.25, 'y': 30.80, 'Mem': 372.85},
    {'method': 'HPINet-S',       'x': 140.45, 'y': 30.92, 'Mem': 445.92},
    {'method': 'SPIN',           'x': 798.44, 'y': 30.98, 'Mem': 441.52},
    {'method': 'ELAN-light',     'x': 42.75,  'y': 30.92, 'Mem': 241.34},
    {'method': 'SwinIR-light',   'x': 177.06, 'y': 30.92, 'Mem': 342.44},
    {'method': 'SRFormer-light', 'x': 145.99, 'y': 31.17, 'Mem': 320.95}
]

colors =  ['#990000',  '#ff0000', '#264653', '#9933FF', '#33CC33',  '#336699', '#FF6666', '#666666', '#FFFF66', '#003366', '#333399', '#000000', '#999933']


def main(save_dir = "plt"):
    for d, color in zip(data, colors):
        area = 10 * d['Mem']    
        ax.scatter(d['x'], d['y'], s=area, alpha=0.8, marker='.', c=color , edgecolors='white', linewidths=2.0)
        font_props = fm.FontProperties('Times New Roman')
        ax.annotate(d['method'], (d['x'], d['y']), xytext=(0, 10), textcoords='offset points',
                    fontproperties=font_props, fontsize=20, ha='right')

     # 网格线
    ax.grid(True)
    # 添加红色的虚线
    ax.plot([data[0]['x'], data[1]['x']], [data[0]['y'], data[1]['y']], 
            color='red', linestyle='--', linewidth=1)

    # 设置 y 轴的范围
    ax.set_ylim(30.25, 31.35)
 
    # 设置 x 轴为对数刻度
    ax.set_xscale('log')
    ax.set_xlim([0, 1e3+100])
 
    # 设置 y 轴的刻度
    ax.set_yticks([30.40, 30.60, 30.80, 31.0, 31.20])
    ax.set_yticklabels([30.40, 30.60, 30.80, 31.0, 31.20], fontproperties=font_props, size=30)

    # 设置轴标签和标题
    ax.set_xlabel('Runtimes (ms)', fontproperties=font_props, fontsize=35)
    ax.set_ylabel('PSNR (dB)', fontproperties=font_props, fontsize=35)
    plt.suptitle('PSNR vs. Runtime and GPU Mem.', fontproperties=font_props, fontsize=35)

    plt.savefig(os.path.join(save_dir, 'model_efficient.pdf'), bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    main()