<script setup lang="ts">
import * as tf from "@tensorflow/tfjs";
import { onMounted, ref } from "vue";
import Plotly, { type PlotData } from "plotly.js-dist-min";
import type { Sequential } from "@tensorflow/tfjs";

// 训练数据
const trainData = {
  sizeMB: [
    0.08, 9.0, 0.001, 0.1, 8.0, 5.0, 0.1, 6.0, 0.05, 0.5, 0.002, 2.0, 0.005,
    10.0, 0.01, 7.0, 6.0, 5.0, 1.0, 1.0,
  ],
  timeSec: [
    0.135, 0.739, 0.067, 0.126, 0.646, 0.435, 0.069, 0.497, 0.068, 0.116, 0.07,
    0.289, 0.076, 0.744, 0.083, 0.56, 0.48, 0.399, 0.153, 0.149,
  ],
};
// 测试数据
const testData = {
  sizeMB: [
    5.0, 0.2, 0.001, 9.0, 0.002, 0.02, 0.008, 4.0, 0.001, 1.0, 0.005, 0.08, 0.8,
    0.2, 0.05, 7.0, 0.005, 0.002, 8.0, 0.008,
  ],
  timeSec: [
    0.425, 0.098, 0.052, 0.686, 0.066, 0.078, 0.07, 0.375, 0.058, 0.136, 0.052,
    0.063, 0.183, 0.087, 0.066, 0.558, 0.066, 0.068, 0.61, 0.057,
  ],
};

// 转化为张量
const trainTensors = {
  sizeMB: tf.tensor2d(trainData.sizeMB, [trainData.sizeMB.length, 1]),
  timeSec: tf.tensor2d(trainData.timeSec, [trainData.timeSec.length, 1]),
};
// 转化为张量
const testTensors = {
  sizeMB: tf.tensor2d(testData.sizeMB, [testData.sizeMB.length, 1]),
  timeSec: tf.tensor2d(testData.timeSec, [testData.timeSec.length, 1]),
};

// 训练损失
let loss = ref<any>(0);

// 验证损失
let valLoss = ref<any>(0);

// 是否训练完成，可以进行预测
let canPred = ref<boolean>(false);

// 输入的文件大小
let input = ref<number>();

// 预测结果
let result = ref<number>();
let model: Sequential;

// 训练数据可视化
const trainP: Partial<PlotData> = {
  name: "训练数据",
  x: trainData.sizeMB,
  y: trainData.timeSec,
  type: "scatter",
  mode: "markers",
  marker: { symbol: "circle", size: 8 },
};

// 测试数据可视化
const testP: Partial<PlotData> = {
  x: testData.sizeMB,
  y: testData.timeSec,
  name: "测试数据",
  mode: "markers",
  type: "scatter",
  marker: { symbol: "triangle-up", size: 10 },
};

// 拟合曲线
const epochsP: Partial<PlotData> = {
  x: [0, 2],
  y: [0, 0.001],
  name: "模型拟合",
  mode: "lines",
  line: { color: "green", width: 1, dash: "dot" },
};

// 更新拟合曲线
const updateScatterWithLines = (
  dataTrace: Partial<PlotData>,
  k: any,
  b: any,
  N: number,
  traceIndex: number
) => {
  dataTrace.x = [0, 10];
  dataTrace.y = [b, b + k * 10];
  var update = {
    x: [dataTrace.x],
    y: [dataTrace.y],
    name: `模型拟合 ${N} epochs`,
  };
  Plotly.restyle("dataSpace", update, traceIndex);
};

// 页面渲染完成，绘制图表
onMounted(() => {
  Plotly.newPlot("dataSpace", [trainP, testP, epochsP], {
    width: 700,
    title: "文件下载时间",
    xaxis: {
      title: "size (MB)",
    },
    yaxis: {
      title: "time (sec)",
    },
  });
});

// 训练
const train = async () => {
  // 创建模型 连续性的模型
  model = tf.sequential();
  updateScatterWithLines(epochsP, 0, 0, 0, 2);
  // 添加神经元
  model.add(tf.layers.dense({ units: 1, inputShape: [1] }));
  model.compile({
    // 损失函数： 均方误差
    loss: tf.losses.meanSquaredError,
    // 优化器： 随机梯度下降
    optimizer: tf.train.sgd(0.005),
  });
  // 训练模型
  await model.fit(trainTensors.sizeMB, trainTensors.timeSec, {
    // 训练的次数
    epochs: 400,
    // 验证数据集
    validationData: [testTensors.sizeMB, testTensors.timeSec],
    callbacks: {
      // 训练过程
      onEpochEnd: async (epoch, logs) => {
        if (logs) {
          loss.value = logs?.loss;
          valLoss.value = (1 - logs?.val_loss) * 100;
          if (epoch % 20 == 0) {
            // 重新绘制拟合曲线
            updateScatterWithLines(
              epochsP,
              model.getWeights()[0].dataSync()[0],
              model.getWeights()[1].dataSync()[0],
              epoch + 20,
              2
            );
          }
        }
      },
    },
  });
  canPred.value = true;
};

// 预测
const pred = () => {
  const pred: any = model.predict(
    tf.tensor([parseFloat((input.value || 0).toString())])
  );
  result.value = pred.dataSync()[0];
};
</script>

<template>
  <main>
    <h3>预测下载时间</h3>
    <div id="dataSpace"></div>
    <el-button type="primary" @click="train">训练</el-button>
    <el-progress v-if="loss" :percentage="loss">
      <el-button type="text">训练损失: {{ loss }}</el-button>
    </el-progress>
    <el-progress v-if="valLoss" :percentage="valLoss">
      <el-button type="text">验证成功率: {{ valLoss }}</el-button>
    </el-progress>
    <div v-if="canPred">
      <el-input
        v-model="input"
        @input="pred"
        placeholder="输入文件大小"
        clearable
      />
      <br />
      <br />
      <el-button type="primary" @click="pred">预测</el-button>
      <br />
      <span v-if="result"
        >文件大小为: {{ input }} MB 需要下载时间为: {{ result }} 秒</span
      >
    </div>
  </main>
</template>
