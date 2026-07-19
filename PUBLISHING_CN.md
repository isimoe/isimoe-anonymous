# ISI-MoE 匿名 GitHub 发布操作指南

这份指南用于将当前项目发布为类似 I2MoE 的 GitHub 开源项目，同时尽量满足双盲审稿的匿名要求。

> 重要：在 AAAI 双盲审稿结束前，不要使用能够关联到作者姓名、单位、个人主页或既有仓库的 GitHub 账号发布项目。

## 一、准备匿名 GitHub 账号

建议新建一个只用于本次审稿的 GitHub 账号或组织。该账号应满足：

- 用户名不包含作者姓名、姓名缩写、实验室或学校名称；
- 使用新的头像或保持默认头像；
- 个人简介、位置、公司和个人网站保持为空；
- 不关注作者的个人账号或实验室组织；
- 不加入能够暴露作者身份的 GitHub 组织；
- 不公开个人邮箱或学校邮箱；
- 不创建与个人仓库之间的 fork 关系。

推荐使用中性的账号名，例如：

```text
anonymous-multimodal
review-artifact-2027
isimoe-review
```

不要直接 fork 个人账号下的原始仓库。GitHub 会公开显示 fork 来源，从而暴露项目与作者账号之间的关系。

## 二、创建匿名仓库

登录匿名 GitHub 账号后：

1. 点击右上角的 `+`；
2. 选择 `New repository`；
3. 仓库名称建议填写 `isimoe-anonymous`；
4. Description 可以填写：

   ```text
   Anonymous implementation and reproducibility artifact for ISI-MoE.
   ```

5. 选择 `Public`；
6. 不要勾选自动创建 README、`.gitignore` 或 License；
7. 点击 `Create repository`。

公开仓库是为了让审稿人无需登录即可访问。项目中不要上传原始数据、患者信息、模型密钥或其他敏感文件。

## 三、配置匿名 Git 提交身份

在项目根目录打开 PowerShell 或终端：

```shell
cd "你的本地项目目录/ISI-MoE"
```

只为当前仓库配置匿名提交身份：

```shell
git config --local user.name "Anonymous Authors"
git config --local user.email "匿名账号的GitHub-no-reply邮箱"
```

GitHub no-reply 邮箱可以在匿名账号的以下位置查看：

```text
Settings → Emails → Keep my email addresses private
```

它通常类似：

```text
123456789+anonymous-login@users.noreply.github.com
```

检查配置：

```shell
git config --local user.name
git config --local user.email
git config --local --list
```

确认输出中没有个人姓名、学校邮箱或单位信息。

不要使用 `--global`，否则会修改电脑上其他仓库的 Git 身份。

## 四、投稿前匿名检查

### 1. 查看准备提交的文件

```shell
git status --short
git ls-files --others --exclude-standard
```

当前项目的 `.gitignore` 已排除：

- `data/`；
- `logs/`；
- `outputs/`；
- `saves/`；
- `.idea/` 和 `.vscode/`；
- Python 缓存；
- 模型权重和检查点；
- 本地网站构建缓存。

### 2. 搜索身份信息

PowerShell 中可以运行：

```powershell
rg -n -i "作者姓名|学校名称|实验室名称|个人用户名|个人邮箱|/home/|Users\\" . `
  -g "!.git/**" `
  -g "!*.zip" `
  -g "!__pycache__/**"
```

请将命令里的中文占位内容替换成所有作者可能出现的真实姓名、拼音、缩写、学校、实验室、GitHub 用户名和邮箱。

第三方开源代码的原作者署名和引用链接不应随意删除；它们通常不是当前投稿作者的身份信息。

### 3. 检查图片和压缩包

重点检查：

- `assets/ISI-MoE.png` 是否包含作者或单位名称；
- PDF 是否包含作者、单位、批注或文档属性；
- ZIP 中是否包含绝对路径、缓存、日志和数据；
- 文件名是否包含姓名、学号、单位或个人用户名。

当前方法图不含作者或单位元数据，但图中上下两个输入均标记为 `M1`，并出现 `Syngy`。如果这不是有意设计，建议投稿前分别确认是否应改成 `M1/M2` 和 `Synergy`。

## 五、创建第一次匿名提交

确认文件无误后执行：

```shell
git add .
git status --short
git diff --cached --stat
git diff --cached
```

仔细检查暂存内容，然后提交：

```shell
git commit -m "Release anonymous ISI-MoE review artifact"
git branch -M main
```

检查提交身份：

```shell
git log -1 --format="%h %an <%ae> %s"
```

预期应看到：

```text
Anonymous Authors <匿名账号的no-reply邮箱>
```

如果已经使用真实姓名或邮箱提交，不要直接推送。应先修正本地提交历史和作者信息，再上传匿名仓库。

## 六、连接匿名 GitHub 仓库

将下面的 `ANONYMOUS_LOGIN` 替换成匿名 GitHub 用户名：

```shell
git remote add origin https://github.com/ANONYMOUS_LOGIN/isimoe-anonymous.git
git remote -v
```

检查远程地址确认它属于匿名账号，然后推送：

```shell
git push -u origin main
```

如果使用浏览器或 GitHub Credential Manager 登录，请确认授权的是匿名账号，不是电脑中已登录的个人账号。

## 七、检查 GitHub 项目主页

推送完成后打开：

```text
https://github.com/ANONYMOUS_LOGIN/isimoe-anonymous
```

确认以下内容可以正常显示：

- README 标题和项目概述；
- ISI-MoE 网络结构图；
- 环境配置说明；
- 数据目录说明；
- 训练命令；
- 复现指南；
- MIT License；
- 匿名审稿说明。

同时检查仓库首页、Insights、Contributors 和 commit 页面，确认没有显示真实作者身份。

## 八、开启 GitHub Pages 网站

当前项目的 Pages 静态网站位于 `docs/` 目录，无需安装额外依赖。

在匿名仓库中执行：

1. 打开 `Settings`；
2. 在左侧找到 `Pages`；
3. 在 `Build and deployment` 下将 Source 设为 `Deploy from a branch`；
4. Branch 选择 `main`；
5. Folder 选择 `/docs`；
6. 点击 `Save`。

发布完成后，网站地址通常是：

```text
https://ANONYMOUS_LOGIN.github.io/isimoe-anonymous/
```

GitHub 第一次部署可能需要几分钟。可以在仓库的 `Actions` 页面查看 Pages 部署状态。

## 九、验证公开网站

使用未登录 GitHub 的无痕浏览器打开 Pages 地址，检查：

- 首页可以访问；
- 样式正常加载；
- 网络结构图正常显示；
- 页面中没有未经验证的数据集或实验结果；
- 下载按钮能够获取匿名 ZIP；
- ZIP 可以正常解压；
- 页面没有作者、单位、邮箱或个人链接；
- 手机端能够正常阅读。

Pages 网站通过 `robots.txt` 和页面元信息请求搜索引擎不要收录，但这不能保证网站绝对不会被发现。真正的匿名性仍取决于账号、仓库历史和页面内容。

## 十、提交 OpenReview 前的最终检查

建议逐项确认：

- [ ] GitHub 账号是全新的匿名账号；
- [ ] 仓库不是从个人仓库 fork；
- [ ] commit 作者和邮箱是匿名信息；
- [ ] README、网站和源码中没有作者身份；
- [ ] `.idea`、缓存、日志、数据和模型权重没有上传；
- [ ] 方法图和 ZIP 元数据已检查；
- [ ] GitHub Pages 可以在未登录状态访问；
- [ ] 下载包与当前仓库代码一致；
- [ ] OpenReview 中只填写匿名仓库或匿名 Pages 地址；
- [ ] 不使用“录用后再公开”作为唯一的可复现证据。

## 十一、审稿结束后

双盲审稿结束且会议政策允许后，可以：

1. 添加作者和单位；
2. 添加论文链接和 BibTeX；
3. 将匿名仓库迁移到正式组织；
4. 在 README 中加入正式引用；
5. 发布正式版本和 Release；
6. 根据数据许可补充更多下载和预处理说明。

在审稿结束前，不建议把匿名仓库转移到作者个人账号，也不要从个人主页、论文预印本或社交媒体链接该匿名仓库。
