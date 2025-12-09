"""Tests for APEE benchmark module."""

import pytest
from apee.benchmarks.datasets import (
    BenchmarkDataset,
    EvaluationScenario,
    TaskCategory,
    Complexity,
)
from apee.benchmarks.runner import BenchmarkConfig, ScenarioResult
from apee.benchmarks.analyzer import BenchmarkAnalyzer, StatisticalAnalysis
from apee.evaluation.quality import QualityScore


class TestBenchmarkDataset:
    """Tests for BenchmarkDataset."""
    
    def test_dataset_loads_scenarios(self):
        """Test that dataset loads default scenarios."""
        dataset = BenchmarkDataset()
        assert len(dataset.scenarios) > 0
    
    def test_dataset_summary(self):
        """Test dataset summary."""
        dataset = BenchmarkDataset()
        summary = dataset.summary()
        
        assert "total_scenarios" in summary
        assert "by_category" in summary
        assert "by_complexity" in summary
        assert summary["total_scenarios"] > 10  # Should have many scenarios
    
    def test_get_by_category(self):
        """Test filtering by category."""
        dataset = BenchmarkDataset()
        
        code_gen = dataset.get_by_category(TaskCategory.CODE_GENERATION)
        assert len(code_gen) > 0
        assert all(s.category == TaskCategory.CODE_GENERATION for s in code_gen)
    
    def test_get_by_complexity(self):
        """Test filtering by complexity."""
        dataset = BenchmarkDataset()
        
        easy = dataset.get_by_complexity(Complexity.EASY)
        assert len(easy) > 0
        assert all(s.complexity == Complexity.EASY for s in easy)
    
    def test_get_by_id(self):
        """Test getting scenario by ID."""
        dataset = BenchmarkDataset()
        
        scenario = dataset.get_by_id("cg_001_fibonacci")
        assert scenario is not None
        assert scenario.id == "cg_001_fibonacci"
        assert scenario.category == TaskCategory.CODE_GENERATION
    
    def test_get_subset(self):
        """Test getting filtered subset."""
        dataset = BenchmarkDataset()
        
        subset = dataset.get_subset(
            categories=[TaskCategory.CODE_GENERATION],
            complexities=[Complexity.EASY],
        )
        
        assert len(subset) > 0
        assert all(
            s.category == TaskCategory.CODE_GENERATION and s.complexity == Complexity.EASY
            for s in subset
        )
    
    def test_add_custom_scenario(self):
        """Test adding custom scenario."""
        dataset = BenchmarkDataset()
        initial_count = len(dataset.scenarios)
        
        custom = EvaluationScenario(
            id="custom_001",
            category=TaskCategory.CODE_GENERATION,
            complexity=Complexity.EASY,
            prompt="Write hello world in Python",
            expected_keywords=["print", "hello"],
        )
        
        dataset.add_scenario(custom)
        assert len(dataset.scenarios) == initial_count + 1
        assert dataset.get_by_id("custom_001") is not None


class TestEvaluationScenario:
    """Tests for EvaluationScenario."""
    
    def test_to_task_conversion(self):
        """Test conversion to APEE Task."""
        scenario = EvaluationScenario(
            id="test_001",
            category=TaskCategory.CODE_GENERATION,
            complexity=Complexity.MEDIUM,
            prompt="Write a function",
            expected_keywords=["def", "return"],
        )
        
        task = scenario.to_task()
        
        assert task.task_id == "test_001"
        assert task.description == "Write a function"
        assert task.complexity == 0.5  # MEDIUM
        assert "category" in task.context
    
    def test_complexity_to_float(self):
        """Test complexity conversion."""
        for complexity, expected in [
            (Complexity.TRIVIAL, 0.1),
            (Complexity.EASY, 0.3),
            (Complexity.MEDIUM, 0.5),
            (Complexity.HARD, 0.7),
            (Complexity.EXPERT, 0.9),
        ]:
            scenario = EvaluationScenario(
                id="test",
                category=TaskCategory.CODE_GENERATION,
                complexity=complexity,
                prompt="test",
            )
            assert scenario._complexity_to_float() == expected


class TestBenchmarkConfig:
    """Tests for BenchmarkConfig."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = BenchmarkConfig()
        
        assert config.models == ["qwen2.5-coder:3b"]
        assert config.runs_per_scenario == 3
        assert config.temperature == 0.7
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = BenchmarkConfig(
            models=["model1", "model2"],
            runs_per_scenario=5,
            categories=[TaskCategory.CODE_GENERATION],
        )
        
        assert len(config.models) == 2
        assert config.runs_per_scenario == 5
        assert config.categories == [TaskCategory.CODE_GENERATION]


class TestBenchmarkAnalyzer:
    """Tests for BenchmarkAnalyzer statistics."""
    
    def test_statistical_analysis(self):
        """Test statistical analysis computation."""
        # Simulate a result with known data
        from apee.benchmarks.runner import BenchmarkResult, ModelStats
        from datetime import datetime
        
        result = BenchmarkResult(
            config=BenchmarkConfig(),
            started_at=datetime.now(),
            completed_at=datetime.now(),
            duration_seconds=10.0,
            scenario_results=[
                ScenarioResult(
                    scenario_id="s1",
                    model="model1",
                    run_number=i,
                    success=True,
                    output="test",
                    latency_ms=100 + i * 10,
                    tokens_used=100,
                    quality_score=QualityScore(overall=0.7 + i * 0.05),
                )
                for i in range(5)
            ],
            model_stats=[
                ModelStats(
                    model="model1",
                    total_scenarios=1,
                    total_runs=5,
                    overall_success_rate=1.0,
                    mean_quality=0.8,
                    std_quality=0.1,
                    mean_latency_ms=120,
                    std_latency_ms=20,
                    mean_tokens=100,
                )
            ],
        )
        
        analyzer = BenchmarkAnalyzer(result)
        analysis = analyzer.get_model_analysis("model1")
        
        assert analysis is not None
        assert analysis.sample_size == 5
        assert 0.7 <= analysis.mean <= 0.95
        assert analysis.min_value >= 0.7
        assert analysis.max_value <= 1.0


class TestDatasetCoverage:
    """Tests for comprehensive dataset coverage."""
    
    def test_has_all_categories(self):
        """Test that dataset covers multiple categories."""
        dataset = BenchmarkDataset()
        
        categories_covered = set(s.category for s in dataset.scenarios)
        
        # Should have at least these core categories
        assert TaskCategory.CODE_GENERATION in categories_covered
        assert TaskCategory.CODE_REVIEW in categories_covered
        assert TaskCategory.REASONING in categories_covered
        assert TaskCategory.QA_FACTUAL in categories_covered
    
    def test_has_all_complexities(self):
        """Test that dataset covers all complexity levels."""
        dataset = BenchmarkDataset()
        
        complexities_covered = set(s.complexity for s in dataset.scenarios)
        
        # Should have easy, medium, and hard at minimum
        assert Complexity.EASY in complexities_covered
        assert Complexity.MEDIUM in complexities_covered
        assert Complexity.HARD in complexities_covered
    
    def test_scenarios_have_expected_keywords(self):
        """Test that most scenarios have expected keywords."""
        dataset = BenchmarkDataset()
        
        with_keywords = sum(1 for s in dataset.scenarios if s.expected_keywords)
        
        # Most scenarios should have expected keywords
        assert with_keywords / len(dataset.scenarios) > 0.8
    
    def test_scenarios_have_rubrics(self):
        """Test that scenarios have scoring rubrics."""
        dataset = BenchmarkDataset()
        
        with_rubrics = sum(1 for s in dataset.scenarios if s.rubric)
        
        # Most scenarios should have rubrics
        assert with_rubrics / len(dataset.scenarios) > 0.7
