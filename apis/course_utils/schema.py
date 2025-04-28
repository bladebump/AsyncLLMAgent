from pydantic import BaseModel, Field

class CourseBaseInfo(BaseModel):
    name: str | None = Field(description="课程名称，例如'网络空间安全基础'", default=None)
    tags: list[int] | None = Field(description="课程标签，为数值列表，描述课程特点的几个关联类目的ID，例如'[1,2,3]'", default=None)
    authors: list[str] | None = Field(description="作者，为字符串列表，例如['张三','李四']", default=None)
    difficulty: str | None = Field(description="课程难度，字符串类型，1（初级）、2（中级）、3（高级），只需要输入数字即可", default=None)
    cover: dict = Field(description="封面图片", default_factory=lambda: {				
        "key": "course_default_cover",				
        "signature": "2239427b7a5db9ddd99e14a10e0a3646",				
        "url": "/adl-oss/resources/course_default_cover?e=1745392405287&fileName&token=SkQYUibnZhlrfZwqMG5x_MBs_lA%3D",				
        "name": "course_default_cover.png",				
        "previewUrl": "/adl-oss/resources/course_default_cover?e=1745392405287&preview=true&token=U5tL-V5dn1Ly-9eTmrLEJQon4Y4%3D",				
        "extension": "png"
    })
    profile: str | None = Field(description="课程简介，简要概括课程内容，例如'本课程主要介绍网络空间安全基础知识'", default=None)

class Chapter(BaseModel):
    name: str | None = Field(description="章节名称，例如'网络安全基础知识'", default=None)
    profile: str | None = Field(description="章节简介，例如'本章节主要介绍网络安全基础知识'", default=None)
    knowledgePointList: list[int] | None = Field(description="知识点ID，这个节下面的所有关联到的知识点id", default=None)
    
class Course(BaseModel):
    baseInfo: CourseBaseInfo = Field(description="课程基本信息")
    chapterList: list[Chapter] | None = Field(description="章节列表，只需要用户输入需要多少总课时即可。", default=None)