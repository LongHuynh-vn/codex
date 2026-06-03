use std::collections::HashMap;

#[derive(Debug, Clone, Default)]
pub(crate) struct SignatureStore {
    calls: HashMap<String, StoredFunctionCall>,
}

#[derive(Debug, Clone)]
pub(crate) struct StoredFunctionCall {
    pub(crate) name: String,
    pub(crate) thought_signature: Option<String>,
}

impl SignatureStore {
    pub(crate) fn record(
        &mut self,
        call_id: impl Into<String>,
        name: impl Into<String>,
        thought_signature: Option<String>,
    ) {
        self.calls.insert(
            call_id.into(),
            StoredFunctionCall {
                name: name.into(),
                thought_signature,
            },
        );
    }

    pub(crate) fn get(&self, call_id: &str) -> Option<&StoredFunctionCall> {
        self.calls.get(call_id)
    }
}
