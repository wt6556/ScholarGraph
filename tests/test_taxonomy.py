"""Taxonomy 模块测试"""

import sys
import os
import unittest
import tempfile
import shutil
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from ScholarGraph.config import (
    Taxonomy,
    Field,
    TaxonomyNotFoundError,
    load_taxonomy,
    get_taxonomy,
    reset_taxonomy
)


class TestField(unittest.TestCase):
    """Field 测试"""

    def test_field_creation(self):
        """测试 Field 创建"""
        print("\n[TEST] test_field_creation")
        field = Field(name="Computer Science", subfields=["AI", "ML"])
        self.assertEqual(field.name, "Computer Science")
        self.assertEqual(len(field.subfields), 2)
        print(f"  - Field: {field.name}, subfields={field.subfields}")


class TestTaxonomy(unittest.TestCase):
    """Taxonomy 测试"""

    def setUp(self):
        """设置测试环境"""
        self.temp_dir = tempfile.mkdtemp()
        reset_taxonomy()
        print(f"\n[SETUP] Created temp dir: {self.temp_dir}")

    def tearDown(self):
        """清理"""
        reset_taxonomy()
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        print(f"[TEARDOWN] Cleaned up")

    def test_taxonomy_creation(self):
        """测试 Taxonomy 创建"""
        print("\n[TEST] test_taxonomy_creation")
        taxonomy = Taxonomy(
            fields=[
                Field(name="CS", subfields=["AI", "ML"]),
                Field(name="Math", subfields=["Stats"])
            ],
            default_field="CS",
            default_subfield="AI"
        )
        self.assertEqual(len(taxonomy.fields), 2)
        self.assertEqual(taxonomy.default_field, "CS")
        print(f"  - Taxonomy: {len(taxonomy.fields)} fields")

    def test_get_field_names(self):
        """测试获取领域名称"""
        print("\n[TEST] test_get_field_names")
        taxonomy = Taxonomy(
            fields=[
                Field(name="CS", subfields=["AI"]),
                Field(name="Math", subfields=["Stats"])
            ]
        )
        names = taxonomy.get_field_names()
        self.assertEqual(len(names), 2)
        self.assertIn("CS", names)
        self.assertIn("Math", names)
        print(f"  - Field names: {names}")

    def test_get_subfields(self):
        """测试获取子领域"""
        print("\n[TEST] test_get_subfields")
        taxonomy = Taxonomy(
            fields=[
                Field(name="CS", subfields=["AI", "ML", "CV"]),
            ]
        )
        subfields = taxonomy.get_subfields("CS")
        self.assertEqual(len(subfields), 3)
        self.assertIn("AI", subfields)
        print(f"  - CS subfields: {subfields}")

    def test_get_subfields_invalid_field(self):
        """测试获取无效领域的子领域"""
        print("\n[TEST] test_get_subfields_invalid_field")
        taxonomy = Taxonomy(fields=[Field(name="CS", subfields=["AI"])])
        subfields = taxonomy.get_subfields("InvalidField")
        self.assertEqual(len(subfields), 0)
        print(f"  - Invalid field subfields: {subfields}")

    def test_is_valid_field(self):
        """测试领域有效性检查"""
        print("\n[TEST] test_is_valid_field")
        taxonomy = Taxonomy(fields=[Field(name="CS", subfields=["AI"])])
        self.assertTrue(taxonomy.is_valid_field("CS"))
        self.assertFalse(taxonomy.is_valid_field("Invalid"))
        print(f"  - CS valid: {taxonomy.is_valid_field('CS')}")

    def test_is_valid_subfield(self):
        """测试子领域有效性检查"""
        print("\n[TEST] test_is_valid_subfield")
        taxonomy = Taxonomy(fields=[Field(name="CS", subfields=["AI", "ML"])])
        self.assertTrue(taxonomy.is_valid_subfield("CS", "AI"))
        self.assertFalse(taxonomy.is_valid_subfield("CS", "Invalid"))
        self.assertFalse(taxonomy.is_valid_subfield("Invalid", "AI"))
        print(f"  - CS/AI valid: {taxonomy.is_valid_subfield('CS', 'AI')}")

    def test_to_llm_prompt_fields(self):
        """测试生成 LLM 提示用字段"""
        print("\n[TEST] test_to_llm_prompt_fields")
        taxonomy = Taxonomy(fields=[
            Field(name="CS", subfields=["AI", "ML"]),
            Field(name="Math", subfields=["Stats"])
        ])
        prompt = taxonomy.to_llm_prompt_fields()
        self.assertIn("CS", prompt)
        self.assertIn("AI", prompt)
        self.assertIn("ML", prompt)
        self.assertIn("Math", prompt)
        print(f"  - LLM prompt fields:\n{prompt}")


class TestTaxonomyLoader(unittest.TestCase):
    """Taxonomy 加载器测试"""

    def setUp(self):
        """设置测试环境"""
        self.temp_dir = tempfile.mkdtemp()
        self.original_dir = os.getcwd()
        os.chdir(self.temp_dir)
        reset_taxonomy()
        print(f"\n[SETUP] Created temp dir: {self.temp_dir}")

    def tearDown(self):
        """清理"""
        os.chdir(self.original_dir)
        reset_taxonomy()
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        print(f"[TEARDOWN] Cleaned up")

    def test_load_taxonomy(self):
        """测试加载分类体系"""
        print("\n[TEST] test_load_taxonomy")
        taxonomy_data = {
            'fields': [
                {'name': 'Computer Science', 'subfields': ['AI', 'ML']},
                {'name': 'Mathematics', 'subfields': ['Stats']}
            ],
            'default_field': 'Computer Science',
            'default_subfield': 'AI'
        }
        taxonomy_path = "./config/taxonomy.yaml"
        os.makedirs("./config", exist_ok=True)
        with open(taxonomy_path, 'w', encoding='utf-8') as f:
            yaml.dump(taxonomy_data, f)

        taxonomy = load_taxonomy(taxonomy_path)
        self.assertEqual(len(taxonomy.fields), 2)
        self.assertEqual(taxonomy.default_field, 'Computer Science')
        print(f"  - Loaded taxonomy: {len(taxonomy.fields)} fields")

    def test_load_taxonomy_not_found(self):
        """测试加载不存在的分类体系"""
        print("\n[TEST] test_load_taxonomy_not_found")
        with self.assertRaises(TaxonomyNotFoundError) as context:
            load_taxonomy("./nonexistent/taxonomy.yaml")
        self.assertIn("不存在", str(context.exception))
        print(f"  - Raised TaxonomyNotFoundError: {context.exception}")

    def test_load_taxonomy_empty_file(self):
        """测试加载空分类体系文件"""
        print("\n[TEST] test_load_taxonomy_empty_file")
        taxonomy_path = "./config/taxonomy.yaml"
        os.makedirs("./config", exist_ok=True)
        with open(taxonomy_path, 'w', encoding='utf-8') as f:
            f.write("")

        with self.assertRaises(TaxonomyNotFoundError) as context:
            load_taxonomy(taxonomy_path)
        self.assertIn("为空", str(context.exception))
        print(f"  - Raised TaxonomyNotFoundError: {context.exception}")

    def test_load_taxonomy_no_fields(self):
        """测试加载没有 fields 的分类体系"""
        print("\n[TEST] test_load_taxonomy_no_fields")
        taxonomy_data = {'default_field': 'CS'}
        taxonomy_path = "./config/taxonomy.yaml"
        os.makedirs("./config", exist_ok=True)
        with open(taxonomy_path, 'w', encoding='utf-8') as f:
            yaml.dump(taxonomy_data, f)

        with self.assertRaises(TaxonomyNotFoundError) as context:
            load_taxonomy(taxonomy_path)
        self.assertIn("无效", str(context.exception))
        print(f"  - Raised TaxonomyNotFoundError: {context.exception}")

    def test_get_taxonomy_singleton(self):
        """测试 taxonomy 单例"""
        print("\n[TEST] test_get_taxonomy_singleton")
        taxonomy_data = {
            'fields': [{'name': 'CS', 'subfields': ['AI']}]
        }
        taxonomy_path = "./config/taxonomy.yaml"
        os.makedirs("./config", exist_ok=True)
        with open(taxonomy_path, 'w', encoding='utf-8') as f:
            yaml.dump(taxonomy_data, f)

        taxonomy1 = get_taxonomy()
        taxonomy2 = get_taxonomy()
        self.assertIs(taxonomy1, taxonomy2)
        print(f"  - Singleton works: {taxonomy1 is taxonomy2}")

    def test_reset_taxonomy(self):
        """测试重置 taxonomy"""
        print("\n[TEST] test_reset_taxonomy")
        taxonomy_data = {
            'fields': [{'name': 'CS', 'subfields': ['AI']}]
        }
        taxonomy_path = "./config/taxonomy.yaml"
        os.makedirs("./config", exist_ok=True)
        with open(taxonomy_path, 'w', encoding='utf-8') as f:
            yaml.dump(taxonomy_data, f)

        taxonomy1 = load_taxonomy(taxonomy_path)
        reset_taxonomy()
        taxonomy2 = load_taxonomy(taxonomy_path)

        self.assertIsNot(taxonomy1, taxonomy2)
        self.assertEqual(taxonomy1.fields[0].name, taxonomy2.fields[0].name)
        print(f"  - Reset works: different instances")


class TestTaxonomyImport(unittest.TestCase):
    """Taxonomy 导入测试"""

    def test_import_all(self):
        """测试导入所有 taxonomy 符号"""
        print("\n[TEST] test_import_all")
        from ScholarGraph.config import (
            Taxonomy,
            Field,
            TaxonomyNotFoundError,
            load_taxonomy,
            get_taxonomy,
            reset_taxonomy
        )
        self.assertIsNotNone(Taxonomy)
        self.assertIsNotNone(Field)
        self.assertIsNotNone(TaxonomyNotFoundError)
        print(f"  - All imports successful")


if __name__ == "__main__":
    print("=" * 60)
    print("Running Taxonomy Tests...")
    print("=" * 60)
    unittest.main(verbosity=2)
