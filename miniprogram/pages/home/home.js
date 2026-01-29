const { API_BASE } = require("../../config")
const PREDICT_URL = API_BASE + "/api/v1/predict/"

Page({
  data: {
    safeTop: 0,
    isUploading: false,
  },

  onLoad() {
    // ✅ 不用 getSystemInfoSync（会有 deprecated 警告）
    let safeTop = 0
    if (wx.getWindowInfo) {
      const w = wx.getWindowInfo()
      safeTop = (w.safeArea && w.safeArea.top) || w.statusBarHeight || 0
    } else {
      const sys = wx.getSystemInfoSync()
      safeTop = (sys.safeAreaInsets && sys.safeAreaInsets.top) || sys.statusBarHeight || 0
    }
    this.setData({ safeTop })
  },

  // 拍照识别
  onTakePhoto() {
    if (this.data.isUploading) return

    wx.chooseMedia({
      count: 1,
      mediaType: ['image'],
      sourceType: ['camera'],
      sizeType: ['compressed'],
      success: (res) => {
        const filePath = (res.tempFiles && res.tempFiles[0] && res.tempFiles[0].tempFilePath) || ""
        if (!filePath) return wx.showToast({ title: '未获取到照片', icon: 'none' })
        this.uploadAndPredict(filePath)
      },
      fail: () => wx.showToast({ title: '已取消拍照', icon: 'none' })
    })
  },

  // 上传图片识别
  onChooseImage() {
    if (this.data.isUploading) return

    wx.chooseMedia({
      count: 1,
      mediaType: ['image'],
      sourceType: ['album'],
      sizeType: ['compressed'],
      success: (res) => {
        const filePath = (res.tempFiles && res.tempFiles[0] && res.tempFiles[0].tempFilePath) || ""
        if (!filePath) return wx.showToast({ title: '未获取到图片', icon: 'none' })
        this.uploadAndPredict(filePath)
      },
      fail: () => wx.showToast({ title: '已取消选择', icon: 'none' })
    })
  },

  // 上传并请求后端推理
  uploadAndPredict(filePath) {
    this.setData({ isUploading: true })
    wx.showLoading({ title: '识别中…', mask: true })

    wx.uploadFile({
      url: PREDICT_URL,
      filePath: filePath,
      name: 'image',     // 后端字段名
      formData: {},
      timeout: 60000,

      success: (res) => {
        console.log("uploadFile statusCode =", res.statusCode)
        console.log("uploadFile raw data =", res.data)

        // 1) HTTP 状态码先判断
        if (res.statusCode !== 200) {
          wx.hideLoading()
          this.setData({ isUploading: false })
          return wx.showToast({ title: '后端错误：' + res.statusCode, icon: 'none' })
        }

        // 2) 解析 JSON
        let data = null
        try {
          data = JSON.parse(res.data)
        } catch (e) {
          wx.hideLoading()
          this.setData({ isUploading: false })
          return wx.showToast({ title: '返回数据解析失败', icon: 'none' })
        }

        // 3) 业务 ok 判断
        if (!data || !data.ok) {
          wx.hideLoading()
          this.setData({ isUploading: false })
          const msg = (data && data.error) ? data.error : '识别失败'
          return wx.showToast({ title: msg, icon: 'none' })
        }

        wx.hideLoading()
        this.setData({ isUploading: false })

        // ✅ 4) 用 eventChannel 把大对象传给 result（最稳）
        wx.navigateTo({
          url: '/pages/result/result?imagePath=' + encodeURIComponent(filePath),
          success: (navRes) => {
            navRes.eventChannel.emit('predictResult', {
              payload: data,
              imagePath: filePath
            })
          },
          fail: () => wx.showToast({ title: '跳转失败', icon: 'none' })
        })
      },

      fail: (err) => {
        console.log("uploadFile fail =", err)
        wx.hideLoading()
        this.setData({ isUploading: false })
        wx.showToast({ title: '网络错误/后端不可达', icon: 'none' })
      }
    })
  }
})
