# 给java接口用，所以命名用小驼峰
from pydantic import BaseModel, Field
from enum import Enum
import json

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
    conditionType: str | None = Field(description="当前阶段的进入条件类型：NONE(无条件)、FRACTION(分值条件)、RANKING(排名条件)。", default=None)
    participantLimit: int | None = Field(description="当前阶段的进入条件为NONE和FRACTION时，表示进入人数和团队数限制，0为不设限制。当进入条件为RANKING时，表示进入排名限制，前N名可以进入", default=None)

class CompetitionStage(BaseModel):
    name: str | None = Field(description="阶段名称，例如'初赛'、'决赛'等", default=None)
    description: str | None = Field(description="阶段描述，详细说明此阶段的内容和目标", default=None)
    point: int | None = Field(description="阶段分值，表示此阶段在整个竞赛中的权重", default=2000)
    startTime: str | None = Field(description="阶段开始时间，格式为'YYYY-MM-DD HH:MM:SS'，一般为4个小时左右", default=None)
    endTime: str | None = Field(description="阶段结束时间，格式为'YYYY-MM-DD HH:MM:SS'，一般为4个小时左右", default=None)
    enterCondition: EnterCondition = Field(description="题目开启的限制条件，决定本阶段题目的出现形式", default_factory=EnterCondition)
    mode: str | None = Field(description="阶段类型，可以是CTF、AWD、BTC或THEORY", default=None)

class CTFGroup(BaseModel):
    name: str | None = Field(description="组名，一般为'WEB'、'RE'、'PWN'，一般和这个组的题目类型相同", default=None)
    corpusId: list[int] | None = Field(description="题库ID，添加方式为告诉需求，比如什么难度几道题", default=None)

class CTFConfig(BaseModel):
    openType: str | None = Field(description="开题方式，ALL(全部开放)、SEQUENCE(按照顺序开放)", default=None)
    canReset: bool | None = Field(description="是否开放选手重制独占靶机，True表示可以重置，False表示不能", default=None)
    resetNum: int | None = Field(description="重制次数，设置允许重置的最大次数，如果不行也要填入0", default=None)

score_data_str = """PY（防止作弊）：
{
     "type": "PY",
     "data": {
            "maxScore": 500,
            "minScore": 200,
            "scoreRate": 10,
     }
}

普通模式(DEFAULT):
{
    "type": "DEFAULT",
    "data": {
        "rewardType": "ratio", // "score" 分数的模式
        "rewardData": {
            "fst": 20,
            "sec": 10,
            "trd": 5,
         }
    }
}

递减模式(RATIO)
{
    "type": "REDUCE",
    "data": {
        "reduceType": "ratio",   // "score" 分数的模式
        "reduce": 10,
        "baseLine": 30
    }
}
"""

class ScorePolicy(BaseModel):
    type: str | None = Field(description="计分方式，PY（防止作弊）、DEFAULT（普通模式）、REDUCE（递减模式）", default=None)
    data: dict | None = Field(description=score_data_str, default=None)

class CTFStage(CompetitionStage):
    mode: ModeType = ModeType.CTF
    answerMode: str | None = Field(description="答题模式，BREAK或者FIX", default=None)
    config: CTFConfig = Field(description="配置，CTF阶段的具体配置", default_factory=CTFConfig)
    scorePolicy: ScorePolicy = Field(description="计分方式，设置此阶段的计分规则", default_factory=ScorePolicy)
    groupList: list[CTFGroup] | None = Field(description="CTF组列表，表示此阶段题目分为哪些组，是一个列表", default=None)

class AWDConfig(BaseModel):
    initPoint: int | None = Field(description="初始分值，设置参赛者的初始分数", default=None)
    roundTime: int | None = Field(description="每轮时长，单位为分钟", default=None)
    isFreeReset: bool | None = Field(description="是否免费重置，True表示可以免费重置，但有次数限制，超出会扣分，False重置不需要扣分", default=None)
    freeResetQty: int | None = Field(description="免费重置次数", default=None)
    resetReduceScore: int | None = Field(description="当不能免费重置的时候，重置需要扣除的分数", default=None)
    resetProtectionTime: int | None = Field(description="重置保护时间，单位为分钟，设置重置后的保护时间", default=None)
    isResettable: bool | None = Field(description="是否开放选手端重制靶机，True表示可以重置，False表示不能", default=None)

class AWDScorePolicyData(BaseModel):
    attackScore: int | None = Field(description="攻击得分", default=None)
    defendScore: int | None = Field(description="防御得分", default=None)
    unavailableScore: int | None = Field(description="不可用得分", default=None)

score_policy_data_str = """
DEFAULT（默认）:
{
    "type": "DEFAULT",
    "data": {
      "attackScore": 15,        //攻击成功得分
      "defendScore": 10,        //被攻击着减分
      "unavailableScore": 5     //服务不可以用减分
    }
}

ZERO_SUM（零和）:
{
    "type": "ZERO_SUM",
    "data": {
        "deductedMaxScore": 0,   // 未知。界面上没有选项，但是传输过程中有
        "serviceScore": 1,       //单个服务可用性分值
        "totalScore": 10         //每题每一轮的分值
    }
}
"""

class AWDScorePolicy(BaseModel):
    type: str | None = Field(description="计分方式，DEFAULT(默认)、ZERO_SUM(零和)", default=None)
    data: dict | None = Field(description=score_policy_data_str, default=None)

class AWDStage(CompetitionStage):
    mode: ModeType = ModeType.AWD
    scorePolicy: AWDScorePolicy = Field(description="计分方式，设置AWD阶段的计分方式", default_factory=AWDScorePolicy)
    config: AWDConfig = Field(description="配置，AWD阶段的具体配置", default_factory=AWDConfig)
    corpusId: list[int] | None = Field(description="题库ID，表示此阶段题目来自哪些题库,AWD一般只有一个题目，类型只有WEB和PWN，仅仅需要用户说明难度和类型即可", default=None)

class BTCScorePolicy(BaseModel):
    additional: bool | None = Field(description="TRUE表示前三通关额外加分,FALSE表示普通积分方式", default=None)
    first: int | None = Field(description="只有在additional为TRUE时有效，第一名得分，如果additional为FALSE，则用0即可", default=None)
    second: int | None = Field(description="只有在additional为TRUE时有效，第二名得分，如果additional为FALSE，则用0即可", default=None)
    third: int | None = Field(description="只有在additional为TRUE时有效，第三名得分，如果additional为FALSE，则用0即可", default=None)

class BTCStage(CompetitionStage):
    mode: ModeType = ModeType.BTC
    scorePolicy: BTCScorePolicy = Field(description="计分规则，设置BTC阶段的计分方式", default_factory=BTCScorePolicy)
    corpusId: list[int] | None = Field(description="题库ID，表示此阶段题目来自哪些题库，类型只有WEB，仅仅需要用户说明难度和数量即可", default=None)

class THEORYConfig(BaseModel):
    isShowAllStem: bool | None = Field(description="是否显示所有题，True表示一次性显示所有题目，False表示逐题显示", default=None)
    isRandomStem: bool | None = Field(description="是否随机出题，True表示随机顺序，False表示固定顺序", default=None)
    canSubmitPaper: bool | None = Field(description="是否开放交卷，True表示可以主动交卷，False表示只能等时间结束", default=None)
    canReviewScore: bool | None = Field(description="是否开放查分，True表示可以查看得分，False表示不能", default=None)
    canReviewPaper: bool | None = Field(description="是否开放查卷，True表示可以查看试卷和答案，False表示不能", default=None)
    canReviewAnalysis: bool | None = Field(description="是否开放查看解析，True表示可以查看解析，False表示不能", default=None)
    paperId: int | None = Field(description="试卷ID，关联到具体的理论试卷", default=None)
    paperName: str | None = Field(description="试卷名称，表示此阶段试卷的名称", default=None)
    mode: str | None = Field(description="答题模式，REGULAR_SCOPE(固定时间开始结束)、REGULAR_SCOPE_TIME(固定答题时长)", default=None)
    useTime: int | None = Field(description="答题时长，单位为分钟。REGULAR_SCOPE_TIME选手答题的时间是受到useTime的限制。如果是REGULAR_SCOPE，则useTime无效", default=None)
    startTime: str | None = Field(description="开始时间，格式为'YYYY-MM-DD HH:MM:SS'", default=None)
    endTime: str | None = Field(description="结束时间，格式为'YYYY-MM-DD HH:MM:SS'", default=None)

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
    stageList: list[CompetitionStage] | None = Field(description="阶段列表，可以创建CTF（夺旗赛）、AWD（攻防赛）、BTC（闯关赛）、THEORY（理论赛）四种阶段，让用户输入要创建一个怎么样的阶段", default=None)
    
    @classmethod
    def model_validate(cls, obj, *args, **kwargs):
        if isinstance(obj, dict) and "stageList" in obj and obj["stageList"]:
            for i, stage in enumerate(obj["stageList"]):
                if isinstance(stage, dict) and "mode" in stage:
                    stage_type = stage["mode"]
                    obj["stageList"][i] = stage_map[stage_type].model_validate(stage)
        return super().model_validate(obj, *args, **kwargs)
        
    def model_dump(self, **kwargs):
        data = super().model_dump(**kwargs)
        if self.stageList:
            data["stageList"] = [stage.model_dump(**kwargs) for stage in self.stageList]
        return data
    
    def model_dump_json(self, **kwargs):
        data = self.model_dump(**kwargs)
        if self.stageList:
            data["stageList"] = [stage.model_dump(**kwargs) for stage in self.stageList]
        return json.dumps(data)
