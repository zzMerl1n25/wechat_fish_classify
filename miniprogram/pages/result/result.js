Page({
  data: {
    safeTop: 0,
    imagePath: "",
    pred: null,
    species: null,
    topkView: []
  },

  onLoad(options) {
    // ✅ 取安全区：优先 getWindowInfo（不会有 deprecated 警告）
    let safeTop = 0
    if (wx.getWindowInfo) {
      const w = wx.getWindowInfo()
      safeTop = (w.safeArea && w.safeArea.top) || w.statusBarHeight || 0
    } else {
      const sys = wx.getSystemInfoSync()
      safeTop = (sys.safeAreaInsets && sys.safeAreaInsets.top) || sys.statusBarHeight || 0
    }
    this.setData({ safeTop })

    // 先放预览图（url 参数只传图片路径，短，不会截断）
    if (options && options.imagePath) {
      this.setData({ imagePath: decodeURIComponent(options.imagePath) })
    }

    // ✅ 从 eventChannel 取 payload（关键）
    const channel = this.getOpenerEventChannel && this.getOpenerEventChannel()
    if (!channel || !channel.on) {
      wx.showToast({ title: '未收到结果数据', icon: 'none' })
      return
    }

    channel.on('predictResult', (evt) => {
      const payload = evt && evt.payload ? evt.payload : null
      const imagePath = evt && evt.imagePath ? evt.imagePath : ""

      if (imagePath) this.setData({ imagePath })

      if (!payload) {
        wx.showToast({ title: '未收到结果数据', icon: 'none' })
        return
      }

      const predRaw = payload.prediction || null
      const speciesRaw = payload.species || null

      // TopK 预处理（WXML 不做 toFixed）
      let topkView = []
      if (predRaw && predRaw.topk && predRaw.topk.length) {
        topkView = predRaw.topk.map((it) => {
          const pct = (typeof it.confidence === 'number')
            ? (it.confidence * 100).toFixed(2) + '%'
            : '—'
          return {
            class_id: it.class_id,
            class_name: it.class_name,
            confidence_pct: pct
          }
        })
      }

      let predView = null
      if (predRaw) {
        predView = {
          model_label: predRaw.model_label || '—',
          confidence_pct: (typeof predRaw.confidence === 'number')
            ? (predRaw.confidence * 100).toFixed(2) + '%'
            : '—'
        }
      }

      const speciesView = speciesRaw ? {
        id: speciesRaw.id || '',
        model_label: speciesRaw.model_label || '',
        name_cn: speciesRaw.name_cn || '—',
        name_en: speciesRaw.name_en || '—',
        scientific_name: speciesRaw.scientific_name || '',
        aliases: speciesRaw.aliases || [],
        features: speciesRaw.features || [],
        description: speciesRaw.description || '暂无',
        habitat: speciesRaw.habitat || '暂无',
        diet: speciesRaw.diet || '暂无',
        created_at: speciesRaw.created_at || '—',
        updated_at: speciesRaw.updated_at || '—'
      } : null

      this.setData({
        pred: predView,
        species: speciesView,
        topkView
      })
    })
  },

  onBack() {
    wx.navigateBack()
  },

  onCopy(e) {
    const text = e.currentTarget.dataset.text || ""
    if (!text) return
    wx.setClipboardData({
      data: text,
      success: () => wx.showToast({ title: '已复制', icon: 'none' })
    })
  }
})
