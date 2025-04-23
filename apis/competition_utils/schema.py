# 给java接口用，所以命名用小驼峰
from pydantic import BaseModel, Field
from enum import Enum

class ModeType(str, Enum):
    CTF = "CTF"
    AWD = "AWD"
    BTC = "BTC"
    THEORY = "THEORY"

class CompetitionBaseInfo(BaseModel):
    name: str | None = Field(description="竞赛名称，例如'网络安全攻防大赛2023'", default=None)
    profile: str | None = Field(description="竞赛简介，简要描述竞赛的目标、特点和意义", default=None)
    punishment: str | None = Field(description="作弊处罚方案，'NOTHING'(无处罚)、'WARNING'(警告)、'DEDUCTION'(扣除所有答题分数)、'BAN'(自动禁赛)", default=None)
    playerOrigin: str | None = Field(description="人员来源，'TEMPORARY_USER'(临时用户)或者'PLATFORM_USER'(平台用户)", default=None)
    entryType: str | None = Field(description="参赛模式，'TEAM'(团队赛)或者'PERSONAL'(个人赛)", default=None)
    maxPlayer: int | None = Field(description="最大参赛人数，设置竞赛可容纳的最大参赛人数", default=None)

class EnterCondition(BaseModel):
    conditionType: str | None = Field(description="当前阶段的进入条件类型：NONE(无条件)、FRACTION(分值条件)、RANKING(排名条件)", default=None)
    participantLimit: int | None = Field(description="当前阶段的进入条件为NONE和FRACTION时，表示进入人数和团队数限制，0为不设限制。当进入条件为RANKING时，表示进入排名限制，前N名可以进入", default=None)

class CompetitionStage(BaseModel):
    name: str | None = Field(description="阶段名称，例如'初赛'、'决赛'等", default=None)
    description: str | None = Field(description="阶段描述，详细说明此阶段的内容和目标", default=None)
    point: int | None = Field(description="阶段分值，表示此阶段在整个竞赛中的权重", default=2000)
    startTime: str | None = Field(description="阶段开始时间，格式为'YYYY-MM-DD HH:MM:SS'", default=None)
    endTime: str | None = Field(description="阶段结束时间，格式为'YYYY-MM-DD HH:MM:SS'", default=None)
    enterCondition: EnterCondition = Field(description="题目开启的限制条件，决定本阶段题目的出现形式", default_factory=EnterCondition)
    mode: str | None = Field(description="阶段类型，可以是CTF、AWD、BTC或THEORY", default=None)

class CTFGroup(BaseModel):
    name: str | None = Field(description="组名，例如'WEB'、'RE'、'PWN'等", default=None)
    corpusId: list[int] | None = Field(description="题库ID，表示此组题目来自哪些题库，是一个列表", default=None)

class CTFConfig(BaseModel):
    openType: str | None = Field(description="开题方式，ALL(全部开放)、SEQUENCE(按照顺序开放)", default=None)
    canReset: bool | None = Field(description="是否开放选手重制独占靶机，True表示可以重置，False表示不能", default=None)
    resetNum: int | None = Field(description="重制次数，设置允许重置的最大次数", default=None)

class ScopeData(BaseModel):
    maxScore: int | None = Field(description="最大分值，设置题目的最高分值", default=None)
    minScore: int | None = Field(description="最小分值，设置题目的最低分值", default=None)
    scoreRate: int | None = Field(description="得分率，影响分值变化的比例", default=None)

class ScopePolicy(BaseModel):
    type: str | None = Field(description="计分方式，PY（防止作弊）、DEFAULT（普通模式）、REDUCE（递减模式）", default=None)
    data: ScopeData = Field(description="数据，包含计分所需的具体数据", default_factory=ScopeData)

class CTFStage(CompetitionStage):
    mode: ModeType = ModeType.CTF
    answerMode: str | None = Field(description="答题模式，BREAK或者FIX", default=None)
    config: CTFConfig = Field(description="配置，CTF阶段的具体配置", default_factory=CTFConfig)
    scopePolicy: ScopePolicy = Field(description="计分方式，设置此阶段的计分规则", default_factory=ScopePolicy)
    groupList: list[CTFGroup] | None = Field(description="CTF组列表，表示此阶段题目分为哪些组", default=None)

class AWDConfig(BaseModel):
    initPoint: int | None = Field(description="初始分值，设置参赛者的初始分数", default=None)
    roundTime: int | None = Field(description="每轮时长，单位为分钟", default=None)
    isFreeReset: bool | None = Field(description="是否免费重置，True表示可以免费重置，False表示需要消耗分数", default=None)
    resetProtectionTime: int | None = Field(description="重置保护时间，单位为分钟，设置重置后的保护时间", default=None)
    isResettable: bool | None = Field(description="是否开放选手端重制靶机，True表示可以重置，False表示不能", default=None)

class AWDStage(CompetitionStage):
    mode: ModeType = ModeType.AWD
    scoreType: str | None = Field(description="计分方式，例如'攻击得分'、'防御得分'、'综合得分'等", default=None)
    config: AWDConfig = Field(description="配置，AWD阶段的具体配置", default_factory=AWDConfig)
    corpusId: list[int] | None = Field(description="题库ID，表示此阶段题目来自哪些题库，是一个列表", default=None)

class BTCScorePolicy(BaseModel):
    additional: bool | None = Field(description="计分方式的额外说明，前三通关额外加分", default=None)

class BTCStage(CompetitionStage):
    mode: ModeType = ModeType.BTC
    scorePolicy: BTCScorePolicy = Field(description="计分规则，设置BTC阶段的计分方式", default_factory=BTCScorePolicy)
    corpusId: list[int] | None = Field(description="题库ID，表示此阶段题目来自哪些题库，是一个列表", default=None)

class THEORYConfig(BaseModel):
    paperId: int | None = Field(description="试卷ID，关联到具体的理论试卷", default=None)
    mode: str | None = Field(description="答题模式，例如'限时作答'、'自由作答'等", default=None)
    isShowAllStem: bool | None = Field(description="是否显示所有题，True表示一次性显示所有题目，False表示逐题显示", default=None)
    isRandomStem: bool | None = Field(description="是否随机出题，True表示随机顺序，False表示固定顺序", default=None)
    canSubmitPaper: bool | None = Field(description="是否开放交卷，True表示可以主动交卷，False表示只能等时间结束", default=None)
    canReviewScore: bool | None = Field(description="是否开放查分，True表示可以查看得分，False表示不能", default=None)
    canReviewPaper: bool | None = Field(description="是否开放查卷，True表示可以查看试卷和答案，False表示不能", default=None)

class THEORYStage(CompetitionStage):
    mode: ModeType = ModeType.THEORY
    config: THEORYConfig = Field(description="配置，THEORY阶段的具体配置", default_factory=THEORYConfig)

stage_map = {
    "CTF": CTFStage,
    "AWD": AWDStage,
    "BTC": BTCStage,
    "THEORY": THEORYStage,
}

class Competition(BaseModel):
    baseInfo: CompetitionBaseInfo = Field(description="竞赛基本信息，包含竞赛名称、简介等基础信息", default_factory=CompetitionBaseInfo)
    stageList: list[CompetitionStage] | None = Field(description="阶段列表，可以创建CTF（夺旗赛）、AWD（攻防赛）、BTC（闯关赛）、THEORY（理论赛）四种阶段", default=None)
    
    @classmethod
    def model_validate(cls, obj, *args, **kwargs):
        if isinstance(obj, dict) and "stageList" in obj and obj["stageList"]:
            for i, stage in enumerate(obj["stageList"]):
                if isinstance(stage, dict) and "mode" in stage:
                    stage_type = stage["mode"]
                    obj["stageList"][i] = stage_map[stage_type].model_validate(stage)
        return super().model_validate(obj, *args, **kwargs)
