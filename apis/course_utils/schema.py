from pydantic import BaseModel, Field
from enum import Enum
import json

class DifficultyType(str, Enum):
    BEGINNER = "1"
    INTERMEDIATE = "2"
    ADVANCED = "3"

class CourseBaseInfo(BaseModel):
    name: str | None = Field(description="课程名称，例如'网络空间安全基础'", default=None)
    tags: list[int] | None = Field(description="课程标签，描述课程特点的几个关联类目的ID，例如'[1,2,3]'", default=None)
    authors: list[str] | None = Field(description="作者，主讲老师或团队名称，例如'张三,李四'", default=None)
    difficulty: DifficultyType | None = Field(description="课程难度，1表示初级，2表示中级，3表示高级", default=None)
    cover: str = Field(description="封面图片", default="1.jpg")
    profile: str | None = Field(description="课程简介，简要概括课程内容，例如'本课程主要介绍网络空间安全基础知识'", default=None)
    knowledgePoints: list[str] | None = Field(description="知识点，课程中涉及到的知识点，例如'网络安全基础知识,网络攻击与防御,网络安全法律法规'", default=None)
    introduction: str | None = Field(description="课程介绍，详细介绍课程内容，例如'本课程主要介绍网络空间安全基础知识，包括网络安全基础知识、网络攻击与防御、网络安全法律法规等'", default=None)
    relatedCourses: list[int] | None = Field(description="相关课程ID，与本课程相关的其他课程的ID，例如'[1,2,3]'", default=None)

class Chapter(BaseModel):
    classHours: int | None = Field(description="课时数，例如48", default=None)
    # name: str | None = Field(description="章名称，例如'网络空间安全概述'", default=None)
    # profile: str | None = Field(description="章简介，简要概括本章内容，例如'介绍网络空间安全的基本概念、重要性，梳理其发展历程，探讨当前面临的主要威胁和挑战。'", default=None)
    # courseld: int | None = Field(description="课程ID，例如'101'", default=None)
    # def model_dump(self, **kwargs):
    #     # 强制所有字段都输出
    #     return {field: getattr(self, field) for field in self.__fields__}

    # def model_dump_json(self, **kwargs):
    #     return json.dumps(self.model_dump(), **kwargs)

class Course(BaseModel):
    baseInfo: CourseBaseInfo = Field(description="课程基本信息，包含课程名称、简介等基础信息", default_factory=CourseBaseInfo)
    chapterList: list[int] | None = Field(description="课时数列表，只需要输入一个数字表示总课时，例如[32]表示32课时", default=None)
    
    @classmethod
    def model_validate(cls, obj, *args, **kwargs):
        if isinstance(obj, dict):
            if "baseInfo" not in obj or obj["baseInfo"] is None:
                obj["baseInfo"] = {}
            if "baseInfo" in obj and obj["baseInfo"] and "classHours" in obj["baseInfo"]:
                class_hours = obj["baseInfo"]["classHours"]
                if class_hours is not None:
                    obj["chapterList"] = [class_hours]
        return super().model_validate(obj, *args, **kwargs)

    def model_dump(self, **kwargs):
        data = super().model_dump(**kwargs)
        return data

    def model_dump_json(self, **kwargs):
        data = self.model_dump(**kwargs)
        return json.dumps(data)