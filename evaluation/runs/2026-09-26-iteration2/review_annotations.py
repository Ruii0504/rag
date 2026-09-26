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
