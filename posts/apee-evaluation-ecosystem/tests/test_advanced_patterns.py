"""
Tests for Advanced APEE Evaluation Patterns.

Tests Calibration Loop, Jury with Personas, and Progressive Deepening implementations.
"""

import pytest
from unittest.mock import MagicMock, patch
import json

from apee.evaluation.advanced_patterns import (
    # Jury Pattern
    JudgePersona,
    JuryEvaluator,
    PersonaConfig,
    PERSONA_CONFIGS,
    # Calibration Loop
    CalibrationLoop,
    CalibratedRubric,
    RubricCriterion,
    # Progressive Deepening
    ProgressiveDeepening,
    ProgressiveResult,
    EvaluationDepth,
    DepthConfig,
    DEFAULT_DEPTH_CONFIGS,
    # Combined
    CalibratedJuryEvaluator,
    # Factories
    create_jury_evaluator,
    create_calibrated_evaluator,
    create_progressive_evaluator,
)
from apee.evaluation.llm_evaluator import (
    EvaluationScore,
    ExecutionTrace,
    CollaborativeTrace,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def sample_execution_trace():
    """Create a sample execution trace for testing."""
    return ExecutionTrace(
        agent_id="test_agent_1",
        agent_role="analyzer",
        task_description="Analyze the given code for potential bugs and suggest improvements.",
        expected_output="A detailed analysis with bug findings and improvement suggestions.",
        final_output="""
## Code Analysis Report

### Bug Findings:
1. **Null Reference**: Line 45 may throw NullPointerException
2. **Resource Leak**: File handle not closed in error path

### Improvement Suggestions:
1. Add input validation for user parameters
2. Use try-with-resources for file handling
3. Consider using Optional instead of null checks

Overall quality: Good with minor issues to address.
""",
        duration_seconds=5.2,
        token_count=450,
    )


@pytest.fixture
def sample_collaborative_trace():
    """Create a sample collaborative trace for testing."""
    return CollaborativeTrace(
        scenario_id="test_scenario_1",
        scenario_description="Multi-agent code review task",
        collaboration_pattern="pipeline",
        participating_agents=["analyzer", "coder", "reviewer"],
        agent_traces=[
            ExecutionTrace(
                agent_id="analyzer_1",
                agent_role="analyzer",
                task_description="Analyze code quality",
                final_output="Analysis complete. Found 2 issues.",
                duration_seconds=3.0,
                token_count=200,
            ),
            ExecutionTrace(
                agent_id="coder_1",
                agent_role="coder",
                task_description="Fix identified issues",
                final_output="Fixed null reference and resource leak.",
                duration_seconds=4.0,
                token_count=300,
            ),
        ],
        final_synthesized_output="Code review complete. 2 bugs fixed.",
        total_duration_seconds=7.0,
    )


@pytest.fixture
def mock_llm_response():
    """Create a mock LLM response factory."""
    def _create_response(score: float, feedback: str):
        return json.dumps({
            "score": score,
            "feedback": feedback,
            "key_observations": ["observation 1"],
            "persona_specific_concerns": ["concern 1"],
        })
    return _create_response


# =============================================================================
# PERSONA TESTS
# =============================================================================

class TestJudgePersonas:
    """Test judge persona configurations."""
    
    def test_all_personas_have_configs(self):
        """All personas should have corresponding configs."""
        for persona in JudgePersona:
            assert persona in PERSONA_CONFIGS
            config = PERSONA_CONFIGS[persona]
            assert config.persona == persona
            assert len(config.system_prompt_modifier) > 50
            assert len(config.focus_areas) > 0
    
    def test_persona_weights_default_to_one(self):
        """Default weights should be 1.0 for equal voting."""
        for config in PERSONA_CONFIGS.values():
            assert config.weight == 1.0
    
    def test_persona_focus_areas_are_unique(self):
        """Each persona should have distinct focus areas."""
        all_focus_areas = []
        for config in PERSONA_CONFIGS.values():
            all_focus_areas.extend(config.focus_areas)
        
        # Check some uniqueness (not completely disjoint, but distinctive)
        assert "error_handling" in PERSONA_CONFIGS[JudgePersona.SKEPTIC].focus_areas
        assert "requirements" in PERSONA_CONFIGS[JudgePersona.LITERALIST].focus_areas
        assert "creativity" in PERSONA_CONFIGS[JudgePersona.OPTIMIST].focus_areas
        assert "practicality" in PERSONA_CONFIGS[JudgePersona.PRAGMATIST].focus_areas
    
    def test_persona_system_prompts_are_distinct(self):
        """Each persona should have a distinct system prompt."""
        prompts = [c.system_prompt_modifier for c in PERSONA_CONFIGS.values()]
        # All prompts should be different
        assert len(set(prompts)) == len(prompts)


# =============================================================================
# JURY EVALUATOR TESTS
# =============================================================================

class TestJuryEvaluator:
    """Test the Jury evaluator with personas."""
    
    def test_jury_initialization_default_personas(self):
        """Jury should initialize with all personas by default."""
        jury = JuryEvaluator(model="test-model")
        assert len(jury.personas) == 4
        assert JudgePersona.SKEPTIC in jury.personas
        assert JudgePersona.LITERALIST in jury.personas
    
    def test_jury_initialization_custom_personas(self):
        """Jury should accept custom persona selection."""
        jury = JuryEvaluator(
            model="test-model",
            personas=[JudgePersona.SKEPTIC, JudgePersona.OPTIMIST],
        )
        assert len(jury.personas) == 2
        assert JudgePersona.LITERALIST not in jury.personas
    
    def test_jury_aggregation_methods(self):
        """Test different aggregation methods."""
        for method in ["mean", "median", "weighted_mean"]:
            jury = JuryEvaluator(model="test-model", aggregation=method)
            assert jury.aggregation == method
    
    def test_jury_evaluate_aggregates_scores(
        self, 
        sample_execution_trace,
        mock_llm_response,
    ):
        """Jury should aggregate scores from all personas."""
        jury = JuryEvaluator(model="test-model", aggregation="mean")
        
        # Mock responses with different scores per persona
        scores = [6.0, 8.0, 9.0, 7.0]
        
        with patch.object(jury, '_call_llm') as mock_call_llm:
            mock_call_llm.side_effect = [
                mock_llm_response(s, f"Feedback for score {s}")
                for s in scores
            ]
            
            result = jury.evaluate(sample_execution_trace)
        
        assert "aggregated_score" in result
        assert "persona_scores" in result
        assert "disagreement" in result
        
        # Mean of [6, 8, 9, 7] = 7.5
        assert result["aggregated_score"].score == 7.5
    
    def test_jury_tracks_disagreement(
        self,
        sample_execution_trace,
        mock_llm_response,
    ):
        """Jury should detect high disagreement between personas."""
        jury = JuryEvaluator(model="test-model")
        
        # High disagreement scenario
        scores = [3.0, 9.0, 5.0, 8.0]  # Range = 6
        
        with patch.object(jury, '_call_llm') as mock_call_llm:
            mock_call_llm.side_effect = [
                mock_llm_response(s, f"Feedback {s}")
                for s in scores
            ]
            
            result = jury.evaluate(sample_execution_trace)
        
        assert result["disagreement"]["high_disagreement"] is True
        assert result["disagreement"]["range"] == 6.0
    
    def test_jury_handles_llm_failures(
        self,
        sample_execution_trace,
    ):
        """Jury should handle individual persona failures gracefully."""
        jury = JuryEvaluator(model="test-model")
        
        # First call fails, rest succeed
        with patch.object(jury, '_call_llm') as mock_call_llm:
            mock_call_llm.side_effect = [
                Exception("LLM failed"),
                json.dumps({"score": 7.0, "feedback": "OK"}),
                json.dumps({"score": 8.0, "feedback": "Good"}),
                json.dumps({"score": 7.5, "feedback": "Fine"}),
            ]
            
            result = jury.evaluate(sample_execution_trace)
        
        # Should still produce result from 3 successful evaluations
        assert result["aggregated_score"].score is not None
        # One persona should have None score
        failed_personas = [
            p for p, data in result["persona_scores"].items()
            if data["score"] is None
        ]
        assert len(failed_personas) == 1


# =============================================================================
# CALIBRATION LOOP TESTS
# =============================================================================

class TestCalibrationLoop:
    """Test the Calibration Loop pattern."""
    
    def test_calibration_requires_multiple_judges(self):
        """Calibration should require at least 2 judges."""
        with pytest.raises(ValueError, match="at least 2 judges"):
            CalibrationLoop(judge_models=["single-model"])
    
    def test_calibration_initialization(self):
        """Calibration should initialize correctly."""
        cal = CalibrationLoop(
            judge_models=["judge1", "judge2"],
            max_calibration_rounds=3,
            agreement_threshold=0.8,
        )
        assert len(cal.judge_models) == 2
        assert cal.max_rounds == 3
        assert cal.agreement_threshold == 0.8
    
    @patch.object(CalibrationLoop, '_call_llm')
    def test_calibration_produces_rubric(self, mock_call_llm):
        """Calibration should produce a valid rubric."""
        # Mock judge proposals
        proposal = json.dumps({
            "criteria": [
                {
                    "name": "accuracy",
                    "description": "How accurate is the output",
                    "weight": 0.5,
                    "score_anchors": {"2": "Wrong", "5": "Partially correct", "8": "Correct"}
                },
                {
                    "name": "clarity",
                    "description": "How clear is the output",
                    "weight": 0.5,
                    "score_anchors": {"2": "Confusing", "5": "OK", "8": "Clear"}
                }
            ],
            "reasoning": "These are important criteria"
        })
        
        # Mock agreement check
        agreement = json.dumps({"agreement": 8, "concerns": "None"})
        
        mock_call_llm.return_value = proposal
        
        cal = CalibrationLoop(judge_models=["judge1", "judge2"])
        
        # Override agreement check to return high agreement
        with patch.object(cal, '_check_agreement', return_value=0.9):
            rubric = cal.calibrate(
                task_description="Test task",
                task_type="test",
            )
        
        assert isinstance(rubric, CalibratedRubric)
        assert len(rubric.criteria) >= 1
        assert rubric.task_type == "test"
    
    def test_rubric_caching(self):
        """Calibration should cache rubrics for same task type."""
        cal = CalibrationLoop(judge_models=["judge1", "judge2"])
        
        # Manually add to cache
        cached_rubric = CalibratedRubric(
            task_type="cached_type",
            criteria=[RubricCriterion(name="test", description="test")],
        )
        cache_key = f"cached_type:{hash('test task'[:100])}"
        cal._rubric_cache[cache_key] = cached_rubric
        
        # Should return cached rubric
        result = cal.calibrate("test task", task_type="cached_type")
        assert result == cached_rubric
    
    def test_default_rubric_fallback(self):
        """Should provide default rubric on calibration failure."""
        cal = CalibrationLoop(judge_models=["judge1", "judge2"])
        
        rubric = cal._default_rubric("fallback_type")
        
        assert rubric.task_type == "fallback_type"
        assert len(rubric.criteria) == 3  # Default has 3 criteria
        assert "calibration failed" in rubric.calibration_notes.lower()


# =============================================================================
# RUBRIC CRITERION TESTS
# =============================================================================

class TestRubricCriterion:
    """Test rubric criterion model."""
    
    def test_criterion_creation(self):
        """Should create valid criterion."""
        criterion = RubricCriterion(
            name="test_criterion",
            description="A test criterion",
            weight=0.5,
            score_anchors={2: "Bad", 5: "OK", 8: "Good"}
        )
        
        assert criterion.name == "test_criterion"
        assert criterion.weight == 0.5
        assert len(criterion.score_anchors) == 3
    
    def test_criterion_weight_bounds(self):
        """Weight should be between 0 and 1."""
        # Valid weight
        RubricCriterion(name="test", description="test", weight=0.5)
        
        # Invalid weights should fail validation
        with pytest.raises(ValueError):
            RubricCriterion(name="test", description="test", weight=-0.1)
        
        with pytest.raises(ValueError):
            RubricCriterion(name="test", description="test", weight=1.5)


# =============================================================================
# CALIBRATED JURY EVALUATOR TESTS
# =============================================================================

class TestCalibratedJuryEvaluator:
    """Test the combined Calibration + Jury evaluator."""
    
    def test_calibrated_jury_initialization(self):
        """Should initialize both calibration and jury components."""
        evaluator = CalibratedJuryEvaluator(
            judge_models=["judge1", "judge2"],
            personas=[JudgePersona.SKEPTIC, JudgePersona.OPTIMIST],
        )
        
        assert evaluator.calibration_loop is not None
        assert evaluator.jury is not None
        assert len(evaluator.jury.personas) == 2
    
    @patch.object(CalibrationLoop, 'calibrate')
    @patch.object(JuryEvaluator, 'evaluate')
    def test_calibrated_jury_workflow(
        self,
        mock_jury_evaluate,
        mock_calibrate,
        sample_execution_trace,
    ):
        """Should run calibration then jury evaluation."""
        # Mock calibration
        mock_rubric = CalibratedRubric(
            task_type="test",
            criteria=[RubricCriterion(name="test", description="test")],
            calibration_rounds=1,
            agreed_by=["judge1", "judge2"],
        )
        mock_calibrate.return_value = mock_rubric
        
        # Mock jury evaluation
        mock_jury_evaluate.return_value = {
            "aggregated_score": EvaluationScore(score=7.5, feedback="Good"),
            "persona_scores": {},
            "disagreement": {"high_disagreement": False},
        }
        
        evaluator = CalibratedJuryEvaluator(judge_models=["judge1", "judge2"])
        result = evaluator.evaluate(sample_execution_trace, task_type="test")
        
        # Should have both calibration and jury results
        assert "calibration" in result
        assert "aggregated_score" in result
        assert result["calibration"]["task_type"] == "test"
        
        # Verify calibrate was called
        mock_calibrate.assert_called_once()
        # Verify jury evaluate was called with rubric
        mock_jury_evaluate.assert_called_once()


# =============================================================================
# FACTORY FUNCTION TESTS
# =============================================================================

class TestFactoryFunctions:
    """Test factory functions."""
    
    def test_create_jury_evaluator_default(self):
        """Should create jury with all personas."""
        jury = create_jury_evaluator(model="test-model")
        assert len(jury.personas) == 4
    
    def test_create_jury_evaluator_custom_personas(self):
        """Should create jury with specified personas."""
        jury = create_jury_evaluator(
            model="test-model",
            personas=["skeptic", "optimist"],
        )
        assert len(jury.personas) == 2
        assert JudgePersona.SKEPTIC in jury.personas
    
    def test_create_calibrated_evaluator(self):
        """Should create calibrated jury evaluator."""
        evaluator = create_calibrated_evaluator(
            judge_models=["judge1", "judge2"],
            personas=["skeptic"],
        )
        assert isinstance(evaluator, CalibratedJuryEvaluator)
        assert len(evaluator.jury.personas) == 1


# =============================================================================
# INTEGRATION TESTS (Requires actual LLM - skip by default)
# =============================================================================

@pytest.mark.skip(reason="Requires running Ollama instance")
class TestIntegration:
    """Integration tests with actual LLM."""
    
    def test_full_jury_evaluation(self, sample_execution_trace):
        """Full jury evaluation with real LLM."""
        jury = create_jury_evaluator(
            model="qwen2.5-coder:3b",
            base_url="http://localhost:11434",
        )
        
        result = jury.evaluate(sample_execution_trace)
        
        assert result["aggregated_score"].score is not None
        assert 0 <= result["aggregated_score"].score <= 10
    
    def test_full_calibrated_evaluation(self, sample_execution_trace):
        """Full calibrated jury evaluation with real LLM."""
        evaluator = create_calibrated_evaluator(
            judge_models=["qwen2.5-coder:7b", "llama3.2:3b"],
            base_url="http://localhost:11434",
        )
        
        result = evaluator.evaluate(sample_execution_trace, task_type="code_analysis")
        
        assert "calibration" in result
        assert "aggregated_score" in result


# =============================================================================
# PROGRESSIVE DEEPENING TESTS
# =============================================================================

class TestEvaluationDepth:
    """Test EvaluationDepth enum and configurations."""
    
    def test_all_depths_exist(self):
        """Should have all 4 depth levels."""
        assert len(EvaluationDepth) == 4
        assert EvaluationDepth.QUICK in EvaluationDepth
        assert EvaluationDepth.STANDARD in EvaluationDepth
        assert EvaluationDepth.DEEP in EvaluationDepth
        assert EvaluationDepth.COMPREHENSIVE in EvaluationDepth
    
    def test_default_configs_for_all_depths(self):
        """Should have default configs for each depth."""
        for depth in EvaluationDepth:
            assert depth in DEFAULT_DEPTH_CONFIGS
            config = DEFAULT_DEPTH_CONFIGS[depth]
            assert isinstance(config, DepthConfig)
            assert config.min_score_to_pass >= 0
            assert config.max_score_to_fail >= 0
    
    def test_depth_configs_have_thresholds(self):
        """Pass threshold should be higher than fail threshold."""
        for depth in [EvaluationDepth.QUICK, EvaluationDepth.STANDARD, EvaluationDepth.DEEP]:
            config = DEFAULT_DEPTH_CONFIGS[depth]
            assert config.min_score_to_pass > config.max_score_to_fail
    
    def test_comprehensive_has_no_early_termination(self):
        """Comprehensive depth should always complete."""
        config = DEFAULT_DEPTH_CONFIGS[EvaluationDepth.COMPREHENSIVE]
        assert config.min_score_to_pass == 0.0
        assert config.max_score_to_fail == 0.0


class TestProgressiveDeepening:
    """Test ProgressiveDeepening evaluator."""
    
    def test_initialization_default(self):
        """Should initialize with default settings."""
        evaluator = ProgressiveDeepening(model="test-model")
        assert evaluator.model == "test-model"
        assert evaluator.max_depth == EvaluationDepth.COMPREHENSIVE
        assert len(evaluator.depth_configs) == 4
    
    def test_initialization_limited_depth(self):
        """Should respect max_depth parameter."""
        evaluator = ProgressiveDeepening(
            model="test-model",
            max_depth=EvaluationDepth.STANDARD,
        )
        assert evaluator.max_depth == EvaluationDepth.STANDARD
    
    def test_quick_evaluation_empty_output(self, sample_execution_trace):
        """Quick check should fail empty outputs."""
        evaluator = ProgressiveDeepening(model="test-model")
        
        # Create trace with empty output
        empty_trace = ExecutionTrace(
            agent_id="test",
            agent_role="test",
            task_description="Test task",
            final_output="",
            duration_seconds=1.0,
            token_count=10,
        )
        
        score = evaluator._quick_evaluation(empty_trace)
        assert score.score == 1.0  # Should fail
        assert "empty" in score.feedback.lower() or "short" in score.feedback.lower()
    
    def test_quick_evaluation_good_output(self, sample_execution_trace):
        """Quick check should pass good outputs."""
        evaluator = ProgressiveDeepening(model="test-model")
        
        score = evaluator._quick_evaluation(sample_execution_trace)
        assert score.score >= 5.0  # Should pass heuristics
        assert "[heuristic]" in score.feedback
    
    def test_quick_evaluation_keyword_matching(self):
        """Quick check should match keywords from task."""
        evaluator = ProgressiveDeepening(model="test-model")
        
        trace = ExecutionTrace(
            agent_id="test",
            agent_role="analyzer",
            task_description="Analyze the authentication system for security vulnerabilities",
            final_output="""
            Security Analysis: The authentication system has been reviewed.
            Found vulnerabilities in password handling and session management.
            Recommendations for improved security measures provided.
            """,
            duration_seconds=2.0,
            token_count=100,
        )
        
        score = evaluator._quick_evaluation(trace)
        assert score.score >= 6.0  # Good keyword coverage
        assert "coverage" in score.feedback.lower()
    
    def test_depth_index_ordering(self):
        """Depth indices should be in correct order."""
        evaluator = ProgressiveDeepening(model="test-model")
        
        assert evaluator._depth_index(EvaluationDepth.QUICK) == 0
        assert evaluator._depth_index(EvaluationDepth.STANDARD) == 1
        assert evaluator._depth_index(EvaluationDepth.DEEP) == 2
        assert evaluator._depth_index(EvaluationDepth.COMPREHENSIVE) == 3
    
    def test_tokens_saved_estimate(self):
        """Should estimate tokens saved correctly."""
        evaluator = ProgressiveDeepening(model="test-model")
        
        # If stopped at QUICK, should save tokens for STANDARD, DEEP, COMPREHENSIVE
        saved = evaluator._estimate_tokens_saved(EvaluationDepth.QUICK)
        assert saved > 0
        
        # If stopped at COMPREHENSIVE, should save nothing
        saved_comprehensive = evaluator._estimate_tokens_saved(EvaluationDepth.COMPREHENSIVE)
        assert saved_comprehensive == 0
    
    def test_fallback_score(self):
        """Should return neutral fallback score."""
        evaluator = ProgressiveDeepening(model="test-model")
        
        score = evaluator._fallback_score()
        assert score.score == 5.0
        assert "fallback" in score.feedback.lower()


class TestProgressiveResult:
    """Test ProgressiveResult dataclass."""
    
    def test_result_creation(self):
        """Should create result with all fields."""
        result = ProgressiveResult(
            final_score=EvaluationScore(score=7.5, feedback="Good"),
            depth_reached=EvaluationDepth.STANDARD,
            early_termination=True,
            termination_reason="pass",
            depth_scores={"quick": 9.0, "standard": 7.5},
            total_time_seconds=2.5,
            tokens_saved_estimate=1500,
        )
        
        assert result.final_score.score == 7.5
        assert result.depth_reached == EvaluationDepth.STANDARD
        assert result.early_termination is True
        assert result.termination_reason == "pass"
        assert len(result.depth_scores) == 2
        assert result.tokens_saved_estimate == 1500


class TestCreateProgressiveEvaluator:
    """Test factory function for progressive evaluator."""
    
    def test_create_default(self):
        """Should create with default settings."""
        evaluator = create_progressive_evaluator(model="test-model")
        assert evaluator.max_depth == EvaluationDepth.COMPREHENSIVE
    
    def test_create_with_max_depth(self):
        """Should respect max_depth string parameter."""
        evaluator = create_progressive_evaluator(
            model="test-model",
            max_depth="standard",
        )
        assert evaluator.max_depth == EvaluationDepth.STANDARD
    
    def test_create_with_custom_thresholds(self):
        """Should apply custom thresholds."""
        evaluator = create_progressive_evaluator(
            model="test-model",
            custom_thresholds={"standard": (7.0, 4.0)},
        )
        
        config = evaluator.depth_configs[EvaluationDepth.STANDARD]
        assert config.min_score_to_pass == 7.0
        assert config.max_score_to_fail == 4.0


@pytest.mark.skip(reason="Requires running Ollama instance")
class TestProgressiveIntegration:
    """Integration tests for Progressive Deepening with actual LLM."""
    
    def test_full_progressive_evaluation(self, sample_execution_trace):
        """Full progressive evaluation with real LLM."""
        evaluator = create_progressive_evaluator(
            model="qwen2.5-coder:3b",
            base_url="http://localhost:11434",
        )
        
        result = evaluator.evaluate(sample_execution_trace)
        
        assert result.final_score.score is not None
        assert 0 <= result.final_score.score <= 10
        assert result.depth_reached in EvaluationDepth
        assert result.total_time_seconds > 0
    
    def test_progressive_early_termination(self):
        """Test that obvious outputs terminate early."""
        evaluator = create_progressive_evaluator(
            model="qwen2.5-coder:3b",
            base_url="http://localhost:11434",
        )
        
        # Create a very poor output that should fail early
        poor_trace = ExecutionTrace(
            agent_id="test",
            agent_role="test",
            task_description="Write a detailed security analysis report",
            final_output="Done.",
            duration_seconds=0.1,
            token_count=5,
        )
        
        result = evaluator.evaluate(poor_trace)
        
        # Should fail at QUICK level
        assert result.early_termination is True
        assert result.termination_reason == "fail"
        assert result.depth_reached == EvaluationDepth.QUICK
