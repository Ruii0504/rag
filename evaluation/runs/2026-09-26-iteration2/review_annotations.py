"""Primary assistant's per-case reviews of final outputs, under frozen rubric v1."""
REVIEWS = {}


def review(number, scores, note, stage="none", *, secondary=(), unsupported=(), factual_errors=(), citations=True):
    case_id = f"RAG-{number:03d}"
    assert case_id not in REVIEWS
    assert len(scores) == 5 and (all(score is None for score in scores) or all(type(score) is int and 1 <= score <= 5 for score in scores))
    REVIEWS[case_id] = {
        "scores": dict(zip(["accuracy", "relevance", "completeness", "fluency", "safety"], scores)),
        "reviewer_note": note, "primary_failure_stage": stage, "secondary_failure_stages": list(secondary),
        "unsupported_claims": list(unsupported), "factual_errors": list(factual_errors),
        "citation_support_passed": citations,
    }


review(1, [5, 5, 5, 5, 5], "覆盖插紧光猫到蓝色WAN口接线及未连接/网线异常含义；原列表序号未再误拦，引用3/6及1/4支持对应事实。")
review(2, [5, 5, 5, 4, 5], "接反原因、正确接法和灯态后果均有引用支持；正确接法与解决方法重复一次，按冻结尺度流畅性扣1分。")
review(3, [None]*5, "改写完成后重排两次均ConnectTimeout，无最终回答；保留首次服务失败，不用重试覆盖。", "format_or_service", citations=False)
review(4, [5, 5, 5, 5, 5], "完整覆盖核对宽带账号密码、欠费和锁定；补充的光猫路由前提及直连测试有明确引用，没有将所有用户都强制改为PPPoE。")
review(5, [5, 5, 5, 5, 5], "不凭打不开后台认定损坏，完整提供连接本机WiFi/LAN、完整管理地址及仍失败重启步骤，引用直接支持。")
review(6, [3, 5, 5, 5, 5], "已提供接本机网络及完整管理地址，但把首次登录的默认SSID写成所有访问都必须使用默认WiFi，并把原文电脑有线LAN路径套给手机；必需条件和适用终端超出了引文。", "generation", secondary=["evidence_judgment", "validation"], unsupported=["手机必须连接默认WiFi或通过网线连接LAN口（将首次登录条件泛化，原引文有线方式适用于电脑）"], citations=False)
review(7, [4, 5, 5, 5, 5], "自动获取IP和可选桥接PPPoE路径完整有据；将原文‘多选用’说成选择后即可直接上网，轻微绝对化，沿用基线此题的非关键不精确扣分尺度。", "generation")
review(8, [5, 5, 5, 5, 5], "开通单、光猫贴纸及遗失联系运营商查询或重置均完整，引用直接支持。")
review(9, [5, 5, 5, 5, 5], "明确检查宽带账号及DNS，引用直接对应Q19；未编造DNS地址。")
review(10, [5, 5, 5, 5, 5], "准确说明未通电或电源故障，并检查适配器及插座，两组事实均有对应引用。")
review(11, [5, 5, 5, 5, 5], "SYS红常亮的系统故障及联系售后完整准确。")
review(12, [5, 5, 5, 5, 5], "说明启动蓝闪正常、无需立刻处理，等待约1分钟至蓝常亮；两处引用支持。")
review(13, [3, 5, 5, 5, 5], "收发数据含义正确，但推导为一定未掉线，并把灯不亮写成掉线必要条件；指示灯表不能排除其他网络故障。", "generation", secondary=["evidence_judgment", "validation"], unsupported=["WAN绿闪并非掉线，只有不亮才表示掉线"], citations=False)
review(14, [5, 5, 5, 5, 5], "灯态解释、后台无线设置分频开关及仅在关闭原因成立时重新开启的条件完整有据。")
review(15, [3, 4, 5, 5, 2], "重连及通知家人完整，但再次额外建议遗忘只能恢复出厂，未区分能否后台改密，也没有清配置后果提醒；与同文Q25冲突，沿用基线此题尺度。", "source_conflict", secondary=["generation", "validation"], factual_errors=["未限定能否登录后台就将忘WiFi密码归结为只能恢复出厂"])
review(16, [3, 4, 5, 5, 2], "核对输入及大小写准确完整，但额外将忘记密码一律导向恢复出厂，遗漏能否登录后台及清配置风险；与015同一冲突按同尺度处理。", "source_conflict", secondary=["generation", "validation"], factual_errors=["未限定能否登录后台就将忘WiFi密码归结为只能恢复出厂"])
review(17, [5, 5, 5, 5, 5], "正常断开、短暂重启生效及用新信息重连均有步骤或FAQ支持。")
review(18, [5, 5, 5, 5, 5], "直接给出数秒至一分钟的生效等待时间，引用Q29准确。")
review(19, [5, 5, 5, 5, 5], "关闭双频合一、分设名称、手动选频及后台路径完整；频段与命名作为示例而非保证，事实有引用支持。")
review(20, [5, 5, 5, 5, 5], "同名及自动选频解释、关闭分设方法完整；虽未命中严格FAQ金标，但正文4和失败处理10直接支持，是基线已登记的等价证据。")
review(21, [5, 5, 4, 5, 5], "主要选择2.4G及穿墙覆盖理由正确，调整建议有据；遗漏金标要求的5G近距离高速对照，不影响主要选择但未完整覆盖。", "generation", secondary=["evidence_judgment", "validation"])
review(22, [5, 5, 5, 5, 5], "确认WiFi6、有线测速区分宽带瓶颈完整；干扰等因素及排查建议有据，不编造确定原因。旧版检索拒答本次恢复。")
review(23, [5, 5, 5, 5, 5], "切5G、关合一、错峰、运营商排查均覆盖；额外的有线测速、信道和干扰源建议有引用支持。")
review(24, [4, 5, 5, 5, 5], "天线位置、频段、干扰及Mesh建议完整，列表未再误拦；末句将‘更稳定’表述成‘保证稳定速率’，为非关键程度绝对化，准确性扣1分。", "generation", secondary=["validation"])
review(25, [5, 5, 5, 5, 5], "金属箱削弱信号、影响散热及移至中心开阔处均覆盖；预埋线与Mesh补充有明确引用，正文3亦支持严格金标遗漏的等价证据组。")
review(26, [5, 5, 5, 5, 5], "避免同位置并远离干扰源的建议正确、有引用。")
review(27, [5, 5, 5, 5, 5], "128台建议值、超限可能影响及减少设备或Mesh扩展完整，保留可能性和实际数量受环境影响的限制。")
review(28, [5, 5, 5, 5, 5], "散热空间及40摄氏度上限两项检查完整，未把烫手当正常温热。")
review(29, [5, 5, 5, 5, 3], "过期、手动关闭、密码及限制均覆盖；仍把主网隔离是否开启放入不能上网排查，未解释隔离只阻断内网访问。未直接要求关闭，但有误导风险，沿用基线安全3分尺度。", "source_conflict", secondary=["generation", "validation"])
review(30, [None]*5, "重排两次ConnectTimeout，未进入证据判定与生成，无最终输出。", "format_or_service", citations=False)
review(31, [5, 5, 5, 4, 5], "绑定状态及网卡/MAC变化完整，随机MAC与换手机建议有据；前两项各自重复同一检查结论，流畅性轻扣。")
review(32, [5, 5, 5, 5, 5], "按设备识别、重新绑定新识别信息及后台操作均获对应FAQ和步骤支持。")
review(33, [5, 5, 5, 5, 5], "说明设备绑定导致不自动转移，重新添加绑定及规则保存步骤完整有据。")
review(34, [5, 5, 5, 5, 5], "明确不支持单App黑名单并提供分类、设备、时段替代范围，否定答案正常通过。")
review(35, [4, 5, 5, 5, 5], "中文SSID支持及改字母数字名称建议完整；将部分旧设备‘可能’兼容性差改成此设备失败的确定原因，非关键确定性过强。", "generation", secondary=["evidence_judgment", "validation"])
review(36, [None]*5, "重排两次ConnectTimeout，无最终回答。", "format_or_service", citations=False)
review(37, [None]*5, "重排两次ConnectTimeout，无最终回答。", "format_or_service", citations=False)
review(38, [None]*5, "重排两次ConnectTimeout，无最终回答；连续三次错误触发保护停止，未补跑覆盖。", "format_or_service", citations=False)
review(39, [5, 5, 5, 5, 5], "最终正确否定上网卡、说明U盘/移动硬盘共享用途，引用直接支持。首稿4G被数字规则拦截后内部修正为上网卡，最终合格但增加一次生成调用。")
review(40, [5, 4, 5, 5, 5], "一主其余子Mesh方案完整准确；旧路由AP/DHCP补充有证据且保留条件，但对两台AX6000的当前场景略有旁支。")
review(41, [None]*5, "重排两次ConnectTimeout，无最终回答。", "format_or_service", citations=False)
review(42, [5, 5, 5, 5, 5], "默认自动与干扰不稳时可尝试手动空闲信道的条件完整，未保证必然改善。")
review(43, [5, 5, 5, 5, 5], "双频标称及实际限制完整；正确保留不能仅凭测速判假的边界，型号核对仅作为排查，未保证真品。首稿否定6000M仍被数字规则误拦，修正后最终合格。")
review(44, [None]*5, "重排两次ConnectTimeout，无最终回答。", "format_or_service", citations=False)
review(45, [5, 5, 5, 5, 5], "明确看机底标签且将nebula2025标为示例，没有声称是实机密码。")
review(46, [5, 5, 5, 5, 5], "登录地址、两频段改密、强密码建议、保存生效及重连均覆盖，文档适用条件有对应引用，沿用基线对操作覆盖的尺度。")
review(47, [3, 5, 3, 5, 5], "可手机配置与管理地址正确，但仍把填写宽带账号写成通用步骤，漏选上网方式及仅PPPoE填运营商凭据的条件。同一引用的Q7已提供准确限定，沿用基线此题尺度。", "generation", secondary=["source_conflict", "evidence_judgment", "validation"])
review(48, [None]*5, "重排两次ConnectTimeout，无最终回答。", "format_or_service", citations=False)
review(49, [5, 5, 5, 4, 5], "光猫LAN到蓝WAN及黄LAN到电脑正确完整，接错后果有据；第三项连续重复相同端口接法，流畅性轻扣。")
review(50, [None]*5, "重排两次ConnectTimeout，无最终回答。", "format_or_service", citations=False)
review(51, [5, 5, 5, 4, 5], "访客网络步骤与独立密码、时长、隔离安全边界完整有据；有效时长示例在步骤和附加说明重复，流畅性轻扣。")
review(52, [5, 5, 5, 5, 5], "绑定设备、19至21时允许上网、保存后效果及识别变化重绑均覆盖，引用支持。")
review(53, [5, 5, 5, 5, 5], "在问题限定‘这款’的语境回答主子合计最多3台，未把3台误作子节点数，引用直接支持。")
review(54, [None]*5, "排查中断后首次完整执行在重排两次ConnectTimeout失败，无法据此判断尺寸语义效果；中断边界见execution-notes。", "format_or_service", citations=False)
review(55, [5, 5, 5, 5, 5], "知识库未给售价，明确不能确认且未编造价格；拒答语句自然。")
review(56, [None]*5, "重排两次ConnectTimeout，无最终回答；无法测得价格与尺寸的部分回答表现。", "format_or_service", citations=False)
review(57, [None]*5, "重排两次ConnectTimeout，无最终回答，双意图答案未生成。", "format_or_service", citations=False)
review(58, [5, 5, 5, 5, 5], "正确承接修改WiFi密码的历史，说明所有设备重连及告知家人，引用支持。")
review(59, [None]*5, "重排两次ConnectTimeout，无最终回答。", "format_or_service", citations=False)
