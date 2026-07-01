# third_party

本目录用于存放 codegen 走读、构建或生成工程时需要参考的第三方源码。

## CMSIS-NN

- 本地路径：`third_party/CMSIS-NN/`
- 来源：Arm 官方 `ARM-software/CMSIS-NN` 仓库 `main` 分支源码包。
- 拉取方式：由于当前环境中 `git clone` 连接 GitHub 超时，本次使用 GitHub codeload tarball 下载并展开。
- License：Apache-2.0，见 `third_party/CMSIS-NN/LICENSE`。
- 当前状态：完整官方源码快照，不包含嵌套 `.git` 历史。
- 维护原则：不得手工修改、裁剪或重排 CMSIS-NN 目录内容；升级时应整体替换为新的官方快照，或改用 submodule / 外部依赖。

后续如果需要长期升级 CMSIS-NN，建议明确选择：

- submodule：改用 Git submodule 记录上游 commit。
- 外部依赖：不提交源码，仅在文档中要求用户提供 `--cmsis-nn-root`。
- vendor 快照：继续提交完整官方源码快照，并在升级时整体替换。
