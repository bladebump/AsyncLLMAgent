# 给java接口用，所以命名用小驼峰
from pydantic import BaseModel, Field
from enum import Enum
import json
from datetime import datetime, timedelta

class ModeType(str, Enum):
    CTF = "CTF"
    AWD = "AWD"
    BTC = "BTC"
    THEORY = "THEORY"

class CompetitionBaseInfo(BaseModel):
    name: str | None = Field(description="竞赛名称，例如'网络安全攻防大赛2025'", default="网络安全攻防大赛2025")
    profile: str | None = Field(description="竞赛简介，简要描述竞赛的目标、特点和意义", default="本竞赛旨在通过实战演练提升参赛者的网络安全攻防能力，涵盖CTF、AWD、BTC等多种竞赛模式，为参赛者提供展示技能、交流学习的平台。")
    punishment: str | None = Field(description="作弊处罚方案，'NOTHING'(无处罚)、'WARNING'(警告)、'DEDUCTION'(扣除所有答题分数)、'BAN'(自动禁赛)", default='BAN')
    playerOrigin: str | None = Field(description="人员来源，'TEMPORARY_USER'(临时用户)或者'PLATFORM_USER'(平台用户)", default='TEMPORARY_USER')
    entryType: str | None = Field(description="参赛模式，'TEAM'(团队赛)或者'PERSONAL'(个人赛)", default='TEAM')
    maxPlayer: int | None = Field(description="最大参赛人数，设置竞赛可容纳的最大参赛人数", default=100)

class EnterCondition(BaseModel):
    conditionType: str | None = Field(description="当前阶段的进入条件类型：NONE(无条件)、FRACTION(分值条件)、RANKING(排名条件)。", default='NONE')
    participantLimit: int | None = Field(description="当前阶段的进入条件为NONE和FRACTION时，表示进入人数和团队数限制，0为不设限制。当进入条件为RANKING时，表示进入排名限制，前N名可以进入", default=0)

class CompetitionStage(BaseModel):
    name: str | None = Field(description="阶段名称，例如'初赛'、'决赛'等", default='大赛-CTF阶段')
    description: str | None = Field(description="阶段描述，详细说明此阶段的内容和目标", default='CTF阶段，主要考察参赛者的基础网络安全知识和攻防能力。')
    point: int | None = Field(description="阶段分值，表示此阶段在整个竞赛中的权重", default=2000)
    startTime: str | None = Field(description="阶段开始时间，格式为'YYYY-MM-DD HH:MM:SS'，一般为4个小时左右", default_factory=lambda: (datetime.now() + timedelta(hours=24)).strftime('%Y-%m-%d %H:%M:%S'))
    endTime: str | None = Field(description="阶段结束时间，格式为'YYYY-MM-DD HH:MM:SS'，一般为4个小时左右", default_factory=lambda: (datetime.now() + timedelta(hours=28)).strftime('%Y-%m-%d %H:%M:%S'))
    enterCondition: EnterCondition = Field(description="题目开启的限制条件，决定本阶段题目的出现形式", default_factory=EnterCondition)
    mode: str | None = Field(description="阶段类型，可以是CTF、AWD、BTC", default='CTF')

class CTFGroup(BaseModel):
    name: str | None = Field(description="组名，一般为'WEB'、'RE'、'PWN'，一般和这个组的题目类型相同", default=None)
    corpusId: list[int] | None = Field(description="题库ID，添加方式为告诉需求，比如什么难度几道题", default=None)

class CTFConfig(BaseModel):
    openType: str | None = Field(description="开题方式，ALL(全部开放)、SEQUENCE(按照顺序开放)", default='ALL')
    canReset: bool | None = Field(description="是否开放选手重制独占靶机，True表示可以重置，False表示不能", default=False)
    resetNum: int | None = Field(description="重制次数，设置允许重置的最大次数，如果不行也要填入0", default=0)

score_data_str = """如果type是PY（防止作弊）：
{
    "maxScore": 500,
    "minScore": 200,
    "scoreRate": 4,
}

如果type是DEFAULT（默认）:
{
    "rewardType": "ratio", // "score" 分数的模式
    "rewardData": {
        "fst": 20,
        "sec": 10,
        "trd": 5,
    }
}

如果type是REDUCE（递减模式）:
{
    "reduceType": "ratio",   // "score" 分数的模式
    "reduce": 10,
    "baseLine": 30
}
"""

class ScorePolicy(BaseModel):
    type: str | None = Field(description="计分方式，PY（防止作弊）、DEFAULT（普通模式）、REDUCE（递减模式）", default='PY')
    data: dict | None = Field(description=score_data_str, default_factory=lambda: {'maxScore': 500, 'minScore': 200, 'scoreRate': 4})

class CTFStage(CompetitionStage):
    mode: ModeType = ModeType.CTF
    answerMode: str | None = Field(description="答题模式，BREAK或者FIX", default='BREAK')
    config: CTFConfig = Field(description="配置，CTF阶段的具体配置", default_factory=CTFConfig)
    scorePolicy: ScorePolicy = Field(description="计分方式，设置此阶段的计分规则", default_factory=ScorePolicy)
    groupList: list[CTFGroup] | None = Field(description="CTF组列表，表示此阶段题目分为哪些组，是一个列表。仅仅引导用户输入['web', 'pwn']这种即可，或者让用户说需要什么组。", default_factory=lambda: [CTFGroup(name='WEB'), CTFGroup(name='MISC'), CTFGroup(name='RE'), CTFGroup(name='PWN')])

class AWDConfig(BaseModel):
    initPoint: int | None = Field(description="初始分值，设置参赛者的初始分数", default=2000)
    roundTime: int | None = Field(description="每轮时长，单位为分钟", default=15)
    isFreeReset: bool | None = Field(description="是否免费重置，True表示可以免费重置，但有次数限制，超出会扣分，False重置不需要扣分", default=True)
    freeResetQty: int | None = Field(description="免费重置次数", default=3)
    resetReduceScore: int | None = Field(description="当不能免费重置的时候，重置需要扣除的分数", default=10)
    resetProtectionTime: int | None = Field(description="重置保护时间，单位为分钟，设置重置后的保护时间", default=3)
    isResettable: bool | None = Field(description="是否开放选手端重制靶机，True表示可以重置，False表示不能", default=True)

score_policy_data_str = """
如果type是DEFAULT（默认）:
{
    "attackScore": 15,        //攻击成功得分
    "defendScore": 10,        //被攻击着减分
    "unavailableScore": 200     //服务不可以用减分
}

如果type是ZERO_SUM（零和）:
{
    "deductedMaxScore": 0,   // 未知。界面上没有选项，但是传输过程中有
    "serviceScore": 200,       //单个服务可用性分值
    "totalScore": 200         //每题每一轮的分值
}
"""

class AWDScorePolicy(BaseModel):
    type: str | None = Field(description="计分方式，DEFAULT(默认)、ZERO_SUM(零和)", default='DEFAULT')
    data: dict | None = Field(description=score_policy_data_str, default_factory=lambda: {'attackScore': 15, 'defendScore': 10, 'unavailableScore': 200})

class AWDStage(CompetitionStage):
    mode: ModeType = ModeType.AWD
    scorePolicy: AWDScorePolicy = Field(description="计分方式，设置AWD阶段的计分方式", default_factory=AWDScorePolicy)
    config: AWDConfig = Field(description="配置，AWD阶段的具体配置", default_factory=AWDConfig)
    corpusId: list[int] | None = Field(description="题库ID，表示此阶段题目来自哪些题库，类型只有WEB和PWN，仅仅需要用户说明难度和类型即可", default=None)

class BTCScorePolicy(BaseModel):
    additional: bool | None = Field(description="TRUE表示前三通关额外加分,FALSE表示普通积分方式", default=True)
    first: int | None = Field(description="只有在additional为TRUE时有效，第一名得分，通常为15，如果additional为FALSE，则用0即可", default=15)
    second: int | None = Field(description="只有在additional为TRUE时有效，第二名得分，通常为10，如果additional为FALSE，则用0即可", default=10)
    third: int | None = Field(description="只有在additional为TRUE时有效，第三名得分，通常为5，如果additional为FALSE，则用0即可", default=5)

class BTCStage(CompetitionStage):
    mode: ModeType = ModeType.BTC
    scorePolicy: BTCScorePolicy = Field(description="计分规则，设置BTC阶段的计分方式", default_factory=BTCScorePolicy)
    corpusId: list[int] | None = Field(description="题库ID，表示此阶段题目来自哪些题库，BTC只有一个题目,类型只有WEB，仅仅需要用户说明难度即可", default=None)

stage_map = {
    "CTF": CTFStage,
    "AWD": AWDStage,
    "BTC": BTCStage,
}

class Competition(BaseModel):
    baseInfo: CompetitionBaseInfo = Field(description="竞赛基本信息，包含竞赛名称、简介等基础信息", default_factory=CompetitionBaseInfo)
    stageList: list[CompetitionStage] | None = Field(description="阶段列表，可以创建CTF（夺旗赛）、AWD（攻防赛）、BTC（闯关赛）三种阶段，让用户输入要创建一个怎么样的阶段", default_factory=lambda: [CTFStage(name='CTF阶段')])
    
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
