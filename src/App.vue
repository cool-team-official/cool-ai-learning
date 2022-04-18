<script setup lang="ts">
import { RouterLink, RouterView } from "vue-router";
import { StarFilled } from "@element-plus/icons-vue";
const handleOpen = (key: string, keyPath: string[]) => {
  console.log(key, keyPath);
};
const handleClose = (key: string, keyPath: string[]) => {
  console.log(key, keyPath);
};

const menuList = [
  {
    name: "线性回归",
    path: "/",
  },
  {
    name: "逻辑回归",
    path: "/",
  },
];
</script>

<template>
  <el-row class="tac">
    <el-col :span="4">
      <el-menu
        default-active="0"
        class="el-menu-vertical-demo"
        @open="handleOpen"
        @close="handleClose"
      >
        <el-menu-item
          v-for="(item, index) in menuList"
          :key="index"
          :index="index.toString()"
        >
          <el-icon><star-filled /></el-icon>
          <RouterLink :to="item.path">{{ item.name }}</RouterLink>
        </el-menu-item>
      </el-menu>
    </el-col>
    <el-col :span="20">
      <div class="content">
        <RouterView />
      </div>
    </el-col>
  </el-row>
</template>

<style>
@import "@/assets/base.css";

.content {
  padding: 20px;
}
/* #app {
  max-width: 1280px;
  margin: 0 auto;
  padding: 2rem;

  font-weight: normal;
} */
</style>
