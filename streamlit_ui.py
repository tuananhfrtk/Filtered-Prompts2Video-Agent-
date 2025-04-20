
import streamlit as st
import pandas as pd
import asyncio
from pathlib import Path
from typing import List, Dict, Any
import logging

from prompts_filtered import PromptAnalyzer
from video_generator_agent import WorkflowManager, Config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class StreamlitUI:
    def __init__(self):
        """Initialize the Streamlit UI components."""
        self.config = Config()
        self.analyzer = PromptAnalyzer()
        self.workflow_manager = WorkflowManager(self.config)
        
    def setup_sidebar(self):
        """Setup the sidebar with configuration options."""
        st.sidebar.title("Configuration")
        
        # Model selection
        self.model = st.sidebar.selectbox(
            "Select Model",
            ["deepseek-r1-distill-llama-70b", "mixtral-8x7b-32768"],
            index=0
        )
        
        # Video settings
        self.video_length = st.sidebar.slider("Video Length (seconds)", 15, 60, 30)
        self.video_style = st.sidebar.selectbox(
            "Video Style",
            ["cinematic", "documentary", "animation", "realistic"],
            index=0
        )
        
        # Batch processing
        self.batch_size = st.sidebar.slider("Batch Size", 1, 10, 3)
    
    def display_prompt_selection(self) -> None:
        """Display prompt selection interface."""
        st.title("AI Video Generator Pipeline")

        harmful_prompts = self.analyzer.get_harmful_prompts_from_db()

        if harmful_prompts.empty:
            st.warning("No harmful prompts found in the database. Please run the prompt analyzer first.")
            return

        selected = st.multiselect(
            "Select prompts to process",
            harmful_prompts["prompt"].tolist(),
            default=harmful_prompts["prompt"].head(3).tolist()
        )

        if selected:
            st.session_state["selected_prompts"] = selected

    
    def display_enhancement_options(self, prompts: List[str]) -> List[str]:
        """Display enhancement options interface."""
        st.subheader("Enhancement Options")

        prompts = st.session_state.get("selected_prompts", []) # pulling
        if not prompts:
            st.info("Please select prompts first.")
            return
        
        enhancement_mode = st.radio(
            "Choose enhancement mode",
            ["Auto-Enhance", "Manual Edit"],
            horizontal=True
        )
        
        if enhancement_mode == "Auto-Enhance":
            if st.button("Enhance Selected Prompts"):
                with st.spinner("Enhancing prompts..."):
                    enhanced_prompts = asyncio.run(self._enhance_prompts(prompts))
                    return enhanced_prompts
        else:
            return self._manual_enhancement_ui(prompts)
            
        return prompts        
        
    async def _enhance_prompts(self, prompts: List[str]) -> None:
        """Enhance prompts using the workflow manager and store/display results."""
        enhanced = []
        for prompt in prompts:
            state = {
                "original_prompt": prompt,
                "refined_prompt": None,
                "script": None,
                "video_url": None,
                "user_feedback": None,
                "needs_edit": False,
                "error": None
            }
            result = await self.workflow_manager._enhance_prompt(state)
            enhanced_prompt = result.get("refined_prompt", prompt)
            enhanced.append(enhanced_prompt)

        # Store in session
        if enhanced:
            st.session_state["enhanced_prompts"] = enhanced #

        # Display results
        st.subheader("Enhanced Prompts")
        for i, ep in enumerate(enhanced):
            st.text_area(f"Enhanced Prompt {i+1}", ep, height=100, key=f"auto_ep_{i}")

        # return enhanced


    def _manual_enhancement_ui(self, prompts: List[str]) -> None:
        """Display manual enhancement interface and store/display results."""
        enhanced = []
        st.subheader("Manual Prompt Editing")

        for i, prompt in enumerate(prompts):
            with st.expander(f"Prompt {i+1}"):
                st.write("Original:", prompt)
                edited = st.text_area(
                    "Edit prompt",
                    prompt,
                    key=f"manual_edit_{i}",
                    height=150
                )
                enhanced.append(edited)

        # Store in session
        if enhanced:
            st.session_state["enhanced_prompts"] = enhanced

        # Optional: display below for summary view
        st.subheader("Edited Prompts Summary")
        for i, ep in enumerate(enhanced):
            st.text_area(f"Edited Prompt {i+1}", ep, height=100, key=f"summary_ep_{i}")

        # return enhanced
    
    def display_video_generation(self, prompts_input: List[str]):
        """Display video generation interface."""
        st.subheader("Video Generation")
        prompts_input = st.session_state.get("enhanced_prompts", [])


        
        if st.button("Generate Videos"):
            progress_bar = st.progress(0)
            status_text = st.empty()

            
            
            results = []
            for i, prompt in enumerate(prompts_input):
                status_text.text(f"Processing prompt {i+1}/{len(prompts_input)}")
                
                try:
                    # Create initial state
                    state = {
                        "original_prompt": prompt,
                        "refined_prompt": prompt,  # Using enhanced prompt
                        "script": None,
                        "video_url": None,
                        "user_feedback": None,
                        "needs_edit": False,
                        "error": None
                    }
                    
                    # Run workflow
                    result = asyncio.run(self.workflow_manager._generate_video(state)
                    )
                    results.append(result)
                    
                except Exception as e:
                    logger.error(f"Error processing prompt: {e}")
                    st.error(f"Error processing prompt: {e}")
                
                progress_bar.progress((i+1) / len(prompts_input))
            
            self._display_results(results)
    
    def _display_results(self, results: List[Dict[str, Any]]):
        """Display the results of video generation."""
        st.subheader("Results")
        
        for i, result in enumerate(results):
            with st.expander(f"Video {i+1}"):
                if result.get("error"):
                    st.error(f"Error: {result['error']}")
                else:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("Original Prompt:")
                        st.write(result["original_prompt"])
                        
                        st.write("Enhanced Prompt:")
                        st.write(result["refined_prompt"])
                    
                    with col2:
                        if result.get("drive_url"):
                            st.video(result["drive_url"])
                            st.download_button(
                                "Download Video",
                                result["drive_url"],
                                key=f"download_{i}"
                            )
        
        # Export results
        if results:
            df = pd.DataFrame(results)
            csv = df.to_csv(index=False)
            st.download_button(
                "Export Results to CSV",
                csv,
                "video_generation_results.csv",
                "text/csv"
            )
    
    def run(self):
        """Run the Streamlit UI."""
        self.setup_sidebar()
        
        # Step 1: Prompt Selection
        selected_prompts = self.display_prompt_selection() #non return with state session
        
        if selected_prompts:
            st.session_state["selected_prompts"] = selected_prompts
        
        # Retrieve selection from session state
        prompts = st.session_state.get("selected_prompts", [])
        #Step 2: 
        if prompts:
            enhanced = self.display_enhancement_options(prompts)

            if enhanced:
                st.session_state["enhanced_prompts"] = enhanced

            prompts_input = st.session_state.get("enhanced_prompts", [])

            if prompts_input:
                results = self.display_video_generation(enhanced)                  

                

def main():
    """Main entry point for the Streamlit UI."""
    ui = StreamlitUI()
    ui.run()

if __name__ == "__main__":
    main()
